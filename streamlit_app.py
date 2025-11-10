# pcos_xai_dashboard_full.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
import shap
from lime.lime_tabular import LimeTabularExplainer
from anchor import anchor_tabular
import warnings

warnings.filterwarnings("ignore")

# ---------- IMPORTANT helper (must match pipeline) ----------
def clip_nonnegative(X):
    import numpy as _np
    X = _np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

# ---------- small SHAP helper ----------
def safe_shap_values_and_array(explainer, X_array, prefer_class=1, use_check_additivity=True):
    """
    Call explainer on X_array and return a 2D numpy array suitable for shap.summary_plot.
    Returns (vals2d, explanation_obj_or_none)
    """
    explanation = None
    try:
        if use_check_additivity:
            try:
                explanation = explainer(X_array, check_additivity=False)
            except TypeError:
                explanation = explainer(X_array)
        else:
            explanation = explainer(X_array)
    except Exception as e:
        # fallback to legacy api if available
        if hasattr(explainer, "shap_values"):
            legacy = explainer.shap_values(X_array)
            if isinstance(legacy, list) and len(legacy) > prefer_class:
                vals = np.array(legacy[prefer_class])
            else:
                vals = np.array(legacy)
            if vals.ndim == 1:
                vals = vals.reshape(1, -1)
            return vals, None
        else:
            raise

    # extract numeric array
    vals = None
    if hasattr(explanation, "values"):
        vals_raw = np.array(explanation.values)
        if vals_raw.ndim == 3:
            if vals_raw.shape[1] > prefer_class:
                vals = vals_raw[:, prefer_class, :]
            else:
                vals = vals_raw[:, -1, :]
        elif vals_raw.ndim == 2:
            vals = vals_raw
        else:
            vals = vals_raw.reshape(vals_raw.shape[0], -1)
    else:
        raw = np.array(explanation)
        if raw.ndim == 3:
            vals = raw[:, prefer_class, :] if raw.shape[1] > prefer_class else raw[:, -1, :]
        elif raw.ndim == 2:
            vals = raw
        elif raw.ndim == 1:
            vals = raw.reshape(1, -1)
        else:
            vals = raw.reshape(raw.shape[0], -1)

    # ensure shape matches X_array
    n_feats = X_array.shape[1]
    if vals.shape[1] != n_feats:
        if vals.shape[1] > n_feats:
            vals = vals[:, :n_feats]
        else:
            pad_width = n_feats - vals.shape[1]
            vals = np.pad(vals, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    return vals, explanation

# ---------- Config ----------
st.set_page_config(page_title="PCOS Predictor + XAI ", layout="wide")
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"

FEATURES = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
TARGET = "pcos_y_n"

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(PIPELINE_FILE) or not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("pipeline or model file missing in working directory")
    preprocessor = joblib.load(PIPELINE_FILE)
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(CLEANED_CSV) if os.path.exists(CLEANED_CSV) else None
    return preprocessor, model, df

try:
    preprocessor, model, df = load_artifacts()
except Exception as e:
    st.error(f"Error loading pipeline/model: {e}")
    st.stop()

# ---------- Global feature selection (shared across tabs) ----------
if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = FEATURES[2]

st.sidebar.header("Global analysis controls")
st.session_state.selected_feature = st.sidebar.selectbox(
    "Select feature to analyze (global)",
    FEATURES,
    index=FEATURES.index(st.session_state.selected_feature)
)

# default input values
defaults = df[FEATURES].median().to_dict() if df is not None else {FEATURES[0]:20.0, FEATURES[1]:1.99, FEATURES[2]:3.5}
st.sidebar.subheader("Input patient features")
beta1 = st.sidebar.number_input("I β-hCG (mIU/mL)", value=float(defaults[FEATURES[0]]))
beta2 = st.sidebar.number_input("II β-hCG (mIU/mL)", value=float(defaults[FEATURES[1]]))
amh = st.sidebar.number_input("AMH (ng/mL)", value=float(defaults[FEATURES[2]]))

sample_df = pd.DataFrame([{FEATURES[0]: beta1, FEATURES[1]: beta2, FEATURES[2]: amh}])[FEATURES]

st.title("PCOS Prediction + Explainable AI ")
st.markdown("Global feature selector is shared across tabs. Selected: **`%s`**" % st.session_state.selected_feature)

# ---------- Prediction ----------
if st.sidebar.button("Predict PCOS"):
    try:
        X_trans = preprocessor.transform(sample_df)
        prob = float(model.predict_proba(X_trans)[:,1][0])
        pred = int(model.predict(X_trans)[0])
    except Exception as e:
        st.error(f"Error during preprocessing/prediction: {e}")
        st.stop()

    col1, col2 = st.columns([2,3])
    with col1:
        st.metric("Predicted Label", "PCOS (1)" if pred==1 else "No PCOS (0)")
        st.write(f"Probability (PCOS=1): **{prob:.4f}**")
        st.progress(min(max(prob,0.0),1.0))
    with col2:
        st.subheader("Input values")
        st.table(sample_df.T.rename(columns={0:"value"}))

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📈 PDP + ICE", "🟢 LIME", "🔵 SHAP", "🟣 ANCHOR", "📊 XAI Comparison", "🎯 Feature Influence"]
    )

    # ---------------- TAB 1: PDP + ICE ----------------
    with tab1:
        st.subheader("PDP + ICE (global & individual effects)")
        if df is None:
            st.warning("Cleaned CSV required for PDP/ICE. Place PCOS_infertility_cleaned.csv in working dir.")
        else:
            X_orig = df[FEATURES]
            try:
                X_pre = preprocessor.transform(X_orig)  # used for sklearn PDP API (preprocessed)
            except Exception as e:
                st.error(f"Could not transform data for PDP: {e}")
                X_pre = None

            pdp_feature = st.session_state.selected_feature
            show_ice = st.checkbox("Show ICE curves", value=True)
            grid_resolution = st.slider("Grid resolution", 20,200,80,10)
            kind = "both" if show_ice else "average"

            if X_pre is not None:
                try:
                    feat_idx = FEATURES.index(pdp_feature)
                    fig, ax = plt.subplots(figsize=(8,4))
                    PartialDependenceDisplay.from_estimator(
                        estimator=model, X=X_pre, features=[feat_idx], kind=kind,
                        grid_resolution=grid_resolution, ax=ax
                    )
                    ax.set_title(f"PDP + ICE for {pdp_feature}")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"PDP compute error: {e}")

            st.markdown("**2D PDP (interaction)**")
            try:
                fig2, ax2 = plt.subplots(figsize=(6,5))
                idx_a = FEATURES.index(pdp_feature)
                idx_b = 0 if idx_a!=0 else 1
                PartialDependenceDisplay.from_estimator(
                    estimator=model, X=X_pre, features=[(idx_a, idx_b)], kind="average",
                    grid_resolution=30, ax=ax2
                )
                ax2.set_title(f"2D PDP: {FEATURES[idx_a]} vs {FEATURES[idx_b]}")
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"2D PDP error: {e}")

    # ---------------- TAB 2: LIME ----------------
    with tab2:
        st.subheader("LIME (local surrogate)")
        try:
            X_train_pre = preprocessor.transform(df[FEATURES])
            lime_explainer = LimeTabularExplainer(
                training_data=np.array(X_train_pre),
                feature_names=FEATURES,
                class_names=["No PCOS","PCOS"],
                mode="classification"
            )
            sample_pre = preprocessor.transform(sample_df)[0]
            exp = lime_explainer.explain_instance(sample_pre, model.predict_proba, num_features=len(FEATURES))
            st.write("Top local contributions (LIME):")
            st.write(exp.as_list())
            st.pyplot(exp.as_pyplot_figure())
        except Exception as e:
            st.error(f"LIME failed: {e}")

    # ---------------- TAB 3: SHAP (final stable version) ----------------
    with tab3:
        st.subheader("SHAP (global + local) — stable mode")

        try:
            # ---- STEP 1: Transform the data exactly as model sees it ----
            X_train_pre = preprocessor.transform(df[FEATURES])
            sample_pre = preprocessor.transform(sample_df)

            # ---- STEP 2: Build a masker directly from transformed NumPy ----
            try:
                masker = shap.maskers.Independent(X_train_pre, max_samples=300)
            except Exception:
                masker = None

            # ---- STEP 3: Build Explainer (robust) ----
            explainer = None
            try:
                if masker is not None:
                    explainer = shap.Explainer(model, masker=masker, model_output="probability")
                else:
                    explainer = shap.Explainer(model, model_output="probability")
                st.caption("SHAP explainer loaded successfully (probability output).")

            except Exception as e_ex:
                st.warning(f"shap.Explainer(...) failed: {e_ex}. Trying TreeExplainer fallback.")
                try:
                    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
                except Exception as e2:
                    st.error(f"Could not build SHAP explainer: {e2}")
                    raise

            # ---- STEP 4: Compute SHAP values safely (sampled) ----
            n_sample = min(300, X_train_pre.shape[0])
            idxs = np.random.RandomState(42).choice(X_train_pre.shape[0], n_sample, replace=False)
            X_sample = X_train_pre[idxs]

            vals_global, expl_obj_global = safe_shap_values_and_array(explainer, X_sample, prefer_class=1, use_check_additivity=True)

            # ---- STEP 5: Global Summary Plot ----
            st.write("### 🌍 Global SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(7, 4))
            shap.summary_plot(vals_global, X_sample, feature_names=FEATURES, show=False)
            st.pyplot(fig)

            # ---- STEP 6: Local Waterfall Plot ----
            st.write("### 🔍 Local SHAP Waterfall for Current Input")
            vals_local, expl_obj_local = safe_shap_values_and_array(explainer, sample_pre, prefer_class=1, use_check_additivity=True)
            v = vals_local.ravel()

            # base value extraction (robust)
            base_val = None
            if expl_obj_local is not None and hasattr(expl_obj_local, "base_values"):
                bv = np.array(expl_obj_local.base_values)
                try:
                    if bv.ndim == 2:
                        base_val = float(bv[0,1]) if bv.shape[1] > 1 else float(bv[0,0])
                    else:
                        base_val = float(bv.ravel()[-1])
                except Exception:
                    base_val = float(bv.ravel()[0])
            else:
                try:
                    base_val = float(model.predict_proba(sample_pre)[:,1][0])
                except Exception:
                    base_val = 0.0

            # try to build Explanation and waterfall
            # --- FIXED waterfall call ---
            # Ensure values are 1D and Explanation is flattened
            expl_for_waterfall = shap.Explanation(
                values=v,  # 1D array of length n_features
                base_values=base_val,
                data=sample_pre[0],
                feature_names=FEATURES
            )

            try:
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                shap.plots.waterfall(expl_for_waterfall, show=False)
                st.pyplot(fig2)
            except Exception as e_w:
                st.warning(f"Waterfall failed ({e_w}) — showing fallback bar chart")
                order = np.argsort(-np.abs(v))
                figb, axb = plt.subplots(figsize=(6, 3))
                axb.bar([FEATURES[i] for i in order], v[order])
                axb.axhline(0, color="k", linewidth=0.6)
                axb.set_title("Local SHAP contributions (fallback)")
                st.pyplot(figb)

        except Exception as e:
            st.error(f"SHAP failed: {e}")

    # ---------------- TAB 4: ANCHOR ----------------
    with tab4:
        st.subheader("ANCHOR (rule-based explanations)")
        try:
            anchor_explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=["No PCOS","PCOS"],
                feature_names=FEATURES,
                train_data=df[FEATURES].values
            )
            exp = anchor_explainer.explain_instance(sample_df.values[0], model.predict, threshold=0.95, delta=0.1)
            rule = " AND ".join(exp.names()) if hasattr(exp, "names") else str(exp)
            precision_val = exp.precision() if callable(getattr(exp,"precision",None)) else getattr(exp,"precision",np.nan)
            coverage_val = exp.coverage() if callable(getattr(exp,"coverage",None)) else getattr(exp,"coverage",np.nan)
            st.write("Anchor rule:")
            st.info(rule)
            st.write(f"Precision: {precision_val:.3f} | Coverage: {coverage_val:.3f}")
        except Exception as e:
            st.error(f"Anchor failed: {e}")

    # ---------------- TAB 5: XAI Comparison (metrics) ----------------
    with tab5:
        st.subheader("XAI Method Comparison (quantitative)")
        if df is None:
            st.warning("Dataset required")
        else:
            X_pre_all = preprocessor.transform(df[FEATURES])
            n_eval = st.slider("Instances to sample for comparison", min_value=20, max_value=min(200, X_pre_all.shape[0]), value=60)
            rng = np.random.RandomState(0)
            idxs = rng.choice(X_pre_all.shape[0], n_eval, replace=False)

            # LIME fidelity (approx by surrogate R2)
            lime_explainer_cmp = LimeTabularExplainer(training_data=np.array(X_pre_all), feature_names=FEATURES, class_names=["No PCOS","PCOS"], mode="classification")
            lime_scores = []
            for i in idxs:
                try:
                    exp_l = lime_explainer_cmp.explain_instance(X_pre_all[i], model.predict_proba, num_features=len(FEATURES))
                    s = getattr(exp_l, "score", None)
                    if s is None:
                        std = np.std(X_pre_all, axis=0, ddof=1)
                        scale = np.maximum(1e-6, 0.1 * std)
                        perturbs = X_pre_all[i].reshape(1,-1) + np.random.normal(0, scale, size=(200, X_pre_all.shape[1]))
                        y_hat = model.predict_proba(perturbs)[:,1]
                        lr = LinearRegression().fit(perturbs, y_hat)
                        s = lr.score(perturbs, y_hat)
                    lime_scores.append(float(s))
                except Exception:
                    lime_scores.append(np.nan)
            lime_mean, lime_std = float(np.nanmean(lime_scores)), float(np.nanstd(lime_scores))

            # SHAP additivity error
            try:
                try:
                    shap_cmp_expl = shap.TreeExplainer(model)
                except Exception:
                    shap_cmp_expl = shap.Explainer(model, masker="tree")
                shap_vals_cmp, _ = safe_shap_values_and_array(shap_cmp_expl, X_pre_all[idxs], prefer_class=1, use_check_additivity=True)
                preds = model.predict_proba(X_pre_all[idxs])[:,1]
                # compute base (best-effort using expected_value)
                ev = getattr(shap_cmp_expl, "expected_value", None)
                if ev is not None:
                    barr = np.atleast_1d(ev)
                    if barr.size > 1:
                        base_pos = np.repeat(float(barr[1]), len(idxs))
                    else:
                        base_pos = np.repeat(float(barr.ravel()[-1]), len(idxs))
                else:
                    base_pos = np.zeros(len(idxs))
                pred_from_shap = np.sum(shap_vals_cmp, axis=1) + base_pos
                shap_errs = np.abs(preds - pred_from_shap)
                shap_mean, shap_std = float(np.mean(shap_errs)), float(np.std(shap_errs))
            except Exception:
                shap_mean, shap_std = np.nan, np.nan

            # ICE variance (for selected feature)
            try:
                feat = st.session_state.selected_feature
                sample_n = min(60, df.shape[0])
                chosen = rng.choice(df.shape[0], sample_n, replace=False)
                grid_points = 20
                grid = np.linspace(df[feat].min(), df[feat].max(), grid_points)
                curves = []
                for r in chosen:
                    row = df.iloc[r].copy()
                    preds_curve = []
                    for val in grid:
                        row2 = row.copy()
                        row2[feat] = val
                        x_t = preprocessor.transform(pd.DataFrame([row2])[FEATURES])
                        preds_curve.append(float(model.predict_proba(x_t)[:,1][0]))
                    curves.append(preds_curve)
                curves = np.array(curves)
                ice_var = float(np.mean(np.var(curves, axis=0)))
            except Exception:
                ice_var = np.nan

            # ANCHOR precision & coverage (sample subset)
            try:
                anchor_cmp = anchor_tabular.AnchorTabularExplainer(class_names=["No PCOS","PCOS"], feature_names=FEATURES, train_data=df[FEATURES].values)
                anchor_idxs = idxs[:min(30, len(idxs))]
                precs, covs = [], []
                for ii in anchor_idxs:
                    try:
                        e = anchor_cmp.explain_instance(df[FEATURES].iloc[ii].values, model.predict, threshold=0.95, delta=0.1)
                        p = e.precision() if callable(getattr(e,"precision",None)) else getattr(e,"precision",np.nan)
                        c = e.coverage() if callable(getattr(e,"coverage",None)) else getattr(e,"coverage",np.nan)
                        precs.append(float(p))
                        covs.append(float(c))
                    except Exception:
                        precs.append(np.nan); covs.append(np.nan)
                prec_mean = float(np.nanmean(precs)) if len(precs)>0 else np.nan
                cov_mean = float(np.nanmean(covs)) if len(covs)>0 else np.nan
            except Exception:
                prec_mean, cov_mean = np.nan, np.nan

            # show table & plot
            df_comp = pd.DataFrame({
                "Method":["LIME fidelity (R2)", "SHAP additivity err (mean)", f"ICE var ({st.session_state.selected_feature})", "Anchor precision (mean)", "Anchor coverage (mean)"],
                "Mean":[lime_mean, shap_mean, ice_var, prec_mean, cov_mean],
                "Std/Notes":[lime_std, shap_std, "-", "-", "-"]
            })
            st.dataframe(df_comp)

            figc, axc = plt.subplots(figsize=(8,3))
            vals_plot = [v if not np.isnan(v) else 0 for v in df_comp["Mean"].values]
            axc.barh(df_comp["Method"], vals_plot)
            axc.set_title("XAI comparison (means)")
            st.pyplot(figc)

    # ---------------- TAB 6: Feature Influence Dashboard ----------------
    with tab6:
        st.subheader("Feature Influence Dashboard (combined per-feature metrics)")

        feats = FEATURES.copy()
        results = {f: {} for f in feats}

        X_pre_all = preprocessor.transform(df[FEATURES])
        n_sample_shap = min(300, X_pre_all.shape[0])
        rng = np.random.RandomState(1)
        shard = rng.choice(X_pre_all.shape[0], n_sample_shap, replace=False)

        # SHAP explainer once (robust)
        try:
            try:
                shap_expl_full = shap.TreeExplainer(model)
            except Exception:
                shap_expl_full = shap.Explainer(model, masker="tree")
            shap_vals_full_arr, _ = safe_shap_values_and_array(shap_expl_full, X_pre_all[shard], prefer_class=1, use_check_additivity=True)
        except Exception:
            shap_vals_full_arr = None

        lime_explainer_all = LimeTabularExplainer(training_data=np.array(X_pre_all), feature_names=FEATURES, class_names=["No PCOS", "PCOS"], mode="classification")

        anchor_expl_all = anchor_tabular.AnchorTabularExplainer(class_names=["No PCOS","PCOS"], feature_names=FEATURES, train_data=df[FEATURES].values)

        for feat in feats:
            # PDP slope (mean response)
            try:
                grid_pts = 20
                grid = np.linspace(df[feat].min(), df[feat].max(), grid_pts)
                mean_preds = []
                for g in grid:
                    X_mod = df[FEATURES].copy()
                    X_mod[feat] = g
                    X_mod_pre = preprocessor.transform(X_mod)
                    preds = model.predict_proba(X_mod_pre)[:,1]
                    mean_preds.append(np.mean(preds))
                lr = LinearRegression().fit(grid.reshape(-1,1), np.array(mean_preds))
                slope = float(lr.coef_[0])
                results[feat]["pdp_abs_slope"] = abs(slope)
            except Exception:
                results[feat]["pdp_abs_slope"] = np.nan

            # ICE variance (mean var across grid)
            try:
                sample_n = min(80, df.shape[0])
                idxs_ice = rng.choice(df.shape[0], sample_n, replace=False)
                grid_pts = 20
                grid = np.linspace(df[feat].min(), df[feat].max(), grid_pts)
                curves = []
                for ii in idxs_ice:
                    row = df.iloc[ii].copy()
                    preds_c = []
                    for g in grid:
                        r2 = row.copy()
                        r2[feat] = g
                        x_t = preprocessor.transform(pd.DataFrame([r2])[FEATURES])
                        preds_c.append(float(model.predict_proba(x_t)[:,1][0]))
                    curves.append(preds_c)
                curves_np = np.array(curves)
                results[feat]["ice_var"] = float(np.mean(np.var(curves_np, axis=0)))
            except Exception:
                results[feat]["ice_var"] = np.nan

            # LIME avg abs weight
            try:
                n_lime = min(80, X_pre_all.shape[0])
                idxs_l = rng.choice(X_pre_all.shape[0], n_lime, replace=False)
                abs_weights = []
                for ii in idxs_l:
                    try:
                        expl = lime_explainer_all.explain_instance(X_pre_all[ii], model.predict_proba, num_features=len(FEATURES))
                        pairs = expl.as_list()
                        found = None
                        for name, w in pairs:
                            if feat in name:
                                found = abs(w); break
                        if found is None:
                            found = 0.0
                        abs_weights.append(float(found))
                    except Exception:
                        abs_weights.append(np.nan)
                results[feat]["lime_avg_abs_w"] = float(np.nanmean(abs_weights))
            except Exception:
                results[feat]["lime_avg_abs_w"] = np.nan

            # SHAP mean abs
            try:
                if shap_vals_full_arr is None:
                    results[feat]["shap_mean_abs"] = np.nan
                else:
                    idx_feat = FEATURES.index(feat)
                    results[feat]["shap_mean_abs"] = float(np.mean(np.abs(shap_vals_full_arr[:, idx_feat])))
            except Exception:
                results[feat]["shap_mean_abs"] = np.nan

            # Anchor presence rate
            try:
                anchor_idxs = rng.choice(df.shape[0], min(80, df.shape[0]), replace=False)
                pres = []
                precis = []
                for ii in anchor_idxs[:40]:
                    try:
                        e = anchor_expl_all.explain_instance(df[FEATURES].iloc[ii].values, model.predict, threshold=0.95, delta=0.1)
                        names = e.names() if hasattr(e, "names") else []
                        name_str = " ".join(names).lower()
                        pres.append(1.0 if feat.lower() in name_str else 0.0)
                        p = e.precision() if callable(getattr(e,"precision",None)) else getattr(e,"precision",np.nan)
                        precis.append(float(p) if p is not None else np.nan)
                    except Exception:
                        pres.append(np.nan); precis.append(np.nan)
                results[feat]["anchor_presence_rate"] = float(np.nanmean(pres))
                results[feat]["anchor_precision_mean"] = float(np.nanmean(precis))
            except Exception:
                results[feat]["anchor_presence_rate"] = np.nan
                results[feat]["anchor_precision_mean"] = np.nan

        # Build DataFrame
        df_feats = pd.DataFrame.from_dict(results, orient="index")
        norm_df = df_feats.copy()
        for col in norm_df.columns:
            colvals = norm_df[col].values.astype(float)
            valid = ~np.isnan(colvals)
            if valid.sum()>0:
                mn, mx = np.nanmin(colvals), np.nanmax(colvals)
                if np.isclose(mx, mn):
                    norm_df[col] = np.where(np.isnan(colvals), np.nan, 1.0)
                else:
                    norm_df[col] = np.where(np.isnan(colvals), np.nan, (colvals - mn) / (mx - mn))
            else:
                norm_df[col] = np.nan

        norm_df["combined_score"] = norm_df.mean(axis=1, skipna=True)

        st.write("Raw per-feature metrics (unsummarized):")
        st.dataframe(df_feats.style.format("{:.4f}"))

        st.write("Normalized metrics and combined score (0-1):")
        st.dataframe(norm_df.style.format("{:.4f}"))

        figc2, axc2 = plt.subplots(figsize=(7,3))
        norm_df_sorted = norm_df.sort_values("combined_score", ascending=True)
        axc2.barh(norm_df_sorted.index, norm_df_sorted["combined_score"], color="tab:blue")
        axc2.set_xlabel("Combined normalized score (0-1)")
        axc2.set_title("Feature Influence Combined Score")
        st.pyplot(figc2)

# ---------- Bottom: Model summary ----------
st.markdown("---")
st.header("Model & Dataset Info")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Model type")
    st.write(type(model).__name__)
    st.subheader("Top level params")
    params = model.get_params()
    show_keys = ["n_estimators","max_depth","class_weight","random_state"]
    st.write({k: params[k] for k in show_keys if k in params})
with c2:
    st.subheader("Dataset summary")
    if df is not None:
        st.write("Shape:", df.shape)
        st.bar_chart(df[TARGET].value_counts())
    else:
        st.write("No cleaned CSV found.")

st.markdown("---")
st.caption("Feature Influence Dashboard aggregates PDP slope, ICE variance, LIME, SHAP and Anchor signals per feature to help compare XAI methods and identify the most important, stable, and agreed-upon features.")
