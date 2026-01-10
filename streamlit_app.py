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

# ---- New: choose whether to explain a manual input or a dataset patient ----
st.sidebar.markdown("---")
st.sidebar.subheader("Input patient source")

# If df exists allow selecting from dataset, otherwise only manual
if df is not None:
    sample_source = st.sidebar.radio("Explain:", ("Manual input", "Select from dataset"))
else:
    sample_source = "Manual input"

# Helper to get default values (from dataset median or fallback)
defaults = df[FEATURES].median().to_dict() if df is not None else {FEATURES[0]:20.0, FEATURES[1]:1.99, FEATURES[2]:3.5}

selected_dataset_index = None
if sample_source == "Select from dataset":
    # prefer an 'id' like column if present for nicer labels
    id_col = None
    if df is not None:
        for candidate in ["id", "patient_id", "ID", "PatientID"]:
            if candidate in df.columns:
                id_col = candidate
                break

    # build options: use index; if id_col exists show index — id in the label
    if df is not None:
        if id_col is not None:
            idx_choice = st.sidebar.selectbox("Choose patient (index — id)", options=list(df.index), format_func=lambda i: f"{i} — {df.loc[i, id_col]}")
            selected_dataset_index = int(idx_choice)
        else:
            idx_choice = st.sidebar.selectbox("Choose patient (index)", options=list(df.index))
            selected_dataset_index = int(idx_choice)

    # show selected patient values (read-only) and also allow using them as manual base
    if selected_dataset_index is not None:
        sel_row = df.loc[selected_dataset_index, FEATURES]
        st.sidebar.markdown("**Selected patient values:**")
        st.sidebar.write(sel_row.to_frame(name="value"))

        # Offer a button to copy dataset values to manual inputs (optional)
        if st.sidebar.button("Use these values (load to manual inputs)"):
            # store loaded values in session so manual controls reflect them below
            st.session_state["_loaded_beta1"] = float(sel_row[FEATURES[0]])
            st.session_state["_loaded_beta2"] = float(sel_row[FEATURES[1]])
            st.session_state["_loaded_amh"] = float(sel_row[FEATURES[2]])

# ---- Manual input controls (shown always so user can edit or preview) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Input patient features (manual)")

# If user loaded dataset values into session_state use those as defaults
def _get_default_feat(feat):
    key_map = {
        FEATURES[0]: "_loaded_beta1",
        FEATURES[1]: "_loaded_beta2",
        FEATURES[2]: "_loaded_amh",
    }
    if key_map[feat] in st.session_state:
        return float(st.session_state[key_map[feat]])
    return float(defaults.get(feat, 0.0))

beta1 = st.sidebar.number_input("I β-hCG (mIU/mL)", value=_get_default_feat(FEATURES[0]))
beta2 = st.sidebar.number_input("II β-hCG (mIU/mL)", value=_get_default_feat(FEATURES[1]))
amh = st.sidebar.number_input("AMH (ng/mL)", value=_get_default_feat(FEATURES[2]))

# Build sample_df depending on source
if sample_source == "Select from dataset" and df is not None and selected_dataset_index is not None:
    # Use the selected dataset row as the sample (guaranteed feature ordering)
    sample_df = df.loc[[selected_dataset_index]][FEATURES].copy().reset_index(drop=True)
else:
    # Use manual inputs
    sample_df = pd.DataFrame([{FEATURES[0]: beta1, FEATURES[1]: beta2, FEATURES[2]: amh}])[FEATURES]

# Show a small badge of which source is being used
st.sidebar.markdown(f"**Active sample source:** `{sample_source}`" + (f" — index {selected_dataset_index}" if selected_dataset_index is not None else ""))

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📈 PDP + ICE", "🟢 LIME", "🔵 SHAP", "🟣 ANCHOR", "📊 XAI Comparison", "🎯 Feature Influence", "🟠 ALE"]
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
            # --- Visualize Anchor rule effect (add inside tab4 after computing `exp`) ---
            import re
            import matplotlib.pyplot as plt
            import seaborn as sns

            # get the rule text (robust)
            try:
                rule_text = " AND ".join(exp.names()) if hasattr(exp, "names") else str(exp)
            except Exception:
                rule_text = str(exp)

            # try to extract numeric threshold for i___beta_hcgmiu_ml
            thresh = None
            m = re.search(r"i___beta_hcgmiu_ml\s*(?:<=|<|>)\s*([0-9.+-eE]+)", rule_text)
            if m:
                try:
                    thresh = float(m.group(1))
                except Exception:
                    thresh = None

            # fallback default (if parsing failed)
            if thresh is None:
                thresh = 1.99

            # Prepare dataset predictions & probabilities
            X_all = df[FEATURES].copy()
            try:
                X_pre_all = preprocessor.transform(X_all)
                probs = model.predict_proba(X_pre_all)[:, 1]
                preds = model.predict(X_pre_all)
            except Exception:
                probs = model.predict_proba(X_all.values)[:, 1]
                preds = model.predict(X_all.values)

            X_all["_prob"] = probs
            X_all["_pred"] = preds

            # compute rule mask & metrics
            mask = X_all["i___beta_hcgmiu_ml"] <= thresh
            coverage_calc = mask.mean()
            if mask.sum() > 0:
                majority_class = X_all.loc[mask, "_pred"].mode().iat[0]
                precision_calc = (X_all.loc[mask, "_pred"] == majority_class).mean()
            else:
                precision_calc = float("nan")

            # Build figure
            fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)

            # Panel 1: histogram
            ax = axes[0]
            sns.histplot(data=X_all, x="i___beta_hcgmiu_ml", hue="_pred", bins=30, multiple="layer", alpha=0.6, ax=ax)
            ax.axvline(thresh, color="black", linestyle="--", linewidth=2, label=f"Anchor threshold = {thresh}")
            ax.set_title("Distribution of I β-hCG by model predicted class (0=no PCOS, 1=PCOS)")
            ax.set_xlabel("i___beta_hcgmiu_ml")
            ax.legend()

            # Panel 2: scatter
            ax = axes[1]
            ax.scatter(X_all["i___beta_hcgmiu_ml"], X_all["_prob"], alpha=0.6, s=18)
            ax.axvline(thresh, color="black", linestyle="--", linewidth=2)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title("Model predicted probability (PCOS=1) vs I β-hCG")
            ax.set_xlabel("i___beta_hcgmiu_ml")
            ax.set_ylabel("Predicted probability (PCOS=1)")

            # Panel 3: precision & coverage bar
            ax = axes[2]
            metrics = {"Coverage (%)": coverage_calc * 100, "Precision (%)": precision_calc * 100}
            names = list(metrics.keys())
            vals = list(metrics.values())
            bars = ax.barh(names, vals, color=["#2a9d8f", "#e76f51"])
            for i, v in enumerate(vals):
                ax.text(v + 1, i, f"{v:.1f}%", va="center")
            ax.set_xlim(0, 105)
            ax.set_title("Anchor rule metrics (computed from dataset & model)")
            ax.set_xlabel("Percent")

            st.pyplot(fig)

            # Show example rows
            st.write(f"Examples (first 10) where rule holds (i___beta_hcgmiu_ml <= {thresh:.2f}):")
            st.dataframe(X_all.loc[mask, FEATURES + ['_pred', '_prob']].head(10))

            st.write("Examples (first 10) where rule does NOT hold:")
            st.dataframe(X_all.loc[~mask, FEATURES + ['_pred', '_prob']].head(10))

        except Exception as e:
            st.error(f"Anchor failed: {e}")

    # ---------------- TAB: XAI Comparison (PDP + ICE + LIME + SHAP + ANCHOR + ALE) ----------------
    with tab5:
        st.subheader("📊 Quantitative Comparison of XAI Methods")

        if df is None:
            st.warning("Dataset required for comparison.")
            st.stop()

        X_pre_all = preprocessor.transform(df[FEATURES])
        rng = np.random.RandomState(42)

        n_eval = st.slider(
            "Number of instances for comparison",
            min_value=20,
            max_value=min(200, X_pre_all.shape[0]),
            value=60
        )

        idxs = rng.choice(X_pre_all.shape[0], n_eval, replace=False)

        # ===================== LIME Fidelity =====================
        lime_explainer = LimeTabularExplainer(
            training_data=np.array(X_pre_all),
            feature_names=FEATURES,
            class_names=["No PCOS", "PCOS"],
            mode="classification"
        )

        lime_scores = []
        for i in idxs:
            try:
                exp = lime_explainer.explain_instance(
                    X_pre_all[i],
                    model.predict_proba,
                    num_features=len(FEATURES)
                )
                lime_scores.append(exp.score)
            except Exception:
                lime_scores.append(np.nan)

        lime_mean = float(np.nanmean(lime_scores))

        # ===================== SHAP Additivity Error =====================
        try:
            shap_explainer = shap.TreeExplainer(model)
            shap_vals = shap_explainer(X_pre_all[idxs])

            vals = shap_vals.values
            if vals.ndim == 3:
                vals = vals[:, 1, :]

            base = shap_explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = base[1]

            preds = model.predict_proba(X_pre_all[idxs])[:, 1]
            preds_shap = np.sum(vals, axis=1) + base
            shap_err = np.mean(np.abs(preds - preds_shap))
        except Exception:
            shap_err = np.nan

        # ===================== ICE Variance =====================
        feat = st.session_state.selected_feature
        grid = np.linspace(df[feat].min(), df[feat].max(), 20)

        ice_curves = []
        for i in idxs[:40]:
            row = df.iloc[i].copy()
            curve = []
            for g in grid:
                row[feat] = g
                pred = model.predict_proba(
                    preprocessor.transform(pd.DataFrame([row])[FEATURES])
                )[:, 1][0]
                curve.append(pred)
            ice_curves.append(curve)

        ice_var = float(np.mean(np.var(np.array(ice_curves), axis=0)))

        # ===================== ANCHOR Precision & Coverage =====================
        anchor_explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=["No PCOS", "PCOS"],
            feature_names=FEATURES,
            train_data=df[FEATURES].values
        )

        precisions, coverages = [], []
        for i in idxs[:25]:
            try:
                exp = anchor_explainer.explain_instance(
                    df[FEATURES].iloc[i].values,
                    model.predict,
                    threshold=0.95
                )
                precisions.append(exp.precision())
                coverages.append(exp.coverage())
            except Exception:
                precisions.append(np.nan)
                coverages.append(np.nan)

        anchor_prec = float(np.nanmean(precisions))
        anchor_cov = float(np.nanmean(coverages))

        # ===================== ALE Mean Absolute Effect (SAFE) =====================
        values = df[feat].values.astype(float)

        # create bins safely
        bins = np.quantile(values, np.linspace(0, 1, 11))
        bins = np.unique(bins)

        ale_effects = []

        if len(bins) > 1:
            for i in range(len(bins) - 1):
                low, high = bins[i], bins[i + 1]
                mask = (values >= low) & (values < high)

                if mask.sum() == 0:
                    continue

                X_low = df.loc[mask, FEATURES].copy()
                X_high = X_low.copy()
                X_low[feat] = low
                X_high[feat] = high

                p_low = model.predict_proba(preprocessor.transform(X_low))[:, 1]
                p_high = model.predict_proba(preprocessor.transform(X_high))[:, 1]

                ale_effects.append(np.mean(p_high - p_low))

        # FINAL safe value
        if len(ale_effects) == 0:
            ale_score = 0.0  # <-- IMPORTANT
        else:
            ale_score = float(np.mean(np.abs(ale_effects)))

        # ===================== RESULTS TABLE =====================
        comp_df = pd.DataFrame({
            "XAI Method": [
                "LIME Fidelity (R²)",
                "SHAP Additivity Error",
                f"ICE Variance ({feat})",
                "ANCHOR Precision",
                "ANCHOR Coverage",
                f"ALE Mean |Effect| ({feat})"
            ],
            "Value": [
                lime_mean,
                shap_err,
                ice_var,
                anchor_prec,
                anchor_cov,
                ale_score
            ]
        })

        numeric_cols = comp_df.select_dtypes(include=[np.number]).columns

        st.dataframe(
            comp_df.style.format(
                {col: "{:.4f}" for col in numeric_cols}
            )
        )

        # ===================== BAR CHART =====================
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(comp_df["XAI Method"], comp_df["Value"])
        ax.set_title("XAI Method Comparison")
        st.pyplot(fig)

        # ===================== INTERPRETATION =====================
        st.info("""
        **How to read this table:**
        - Higher LIME fidelity → better local approximation
        - Lower SHAP additivity error → more consistent explanations
        - Lower ICE variance → more stable individual behavior
        - Higher Anchor precision → more reliable rules
        - Higher ALE effect → stronger, correlation-aware influence
        """)

    # ---------------- TAB 6: Feature Influence Dashboard (WITH ALE) ----------------
    with tab6:
        st.subheader("🎯 Feature Influence Dashboard (PDP + ICE + LIME + SHAP + ANCHOR + ALE)")

        feats = FEATURES.copy()
        results = {f: {} for f in feats}

        X_pre_all = preprocessor.transform(df[FEATURES])
        rng = np.random.RandomState(1)

        # ---------- SHAP (once) ----------
        try:
            shap_expl = shap.TreeExplainer(model)
            shap_vals = shap_expl(X_pre_all)
            vals = shap_vals.values
            if vals.ndim == 3:
                vals = vals[:, 1, :]
        except Exception:
            vals = None

        # ---------- LIME ----------
        lime_expl = LimeTabularExplainer(
            training_data=np.array(X_pre_all),
            feature_names=FEATURES,
            class_names=["No PCOS", "PCOS"],
            mode="classification"
        )

        # ---------- ANCHOR ----------
        anchor_expl = anchor_tabular.AnchorTabularExplainer(
            class_names=["No PCOS", "PCOS"],
            feature_names=FEATURES,
            train_data=df[FEATURES].values
        )

        for feat in feats:
            # ================= PDP SLOPE =================
            try:
                grid = np.linspace(df[feat].min(), df[feat].max(), 20)
                mean_preds = []
                for g in grid:
                    X_mod = df[FEATURES].copy()
                    X_mod[feat] = g
                    preds = model.predict_proba(preprocessor.transform(X_mod))[:, 1]
                    mean_preds.append(np.mean(preds))
                lr = LinearRegression().fit(grid.reshape(-1, 1), mean_preds)
                results[feat]["PDP |slope|"] = abs(lr.coef_[0])
            except Exception:
                results[feat]["PDP |slope|"] = np.nan

            # ================= ICE VARIANCE =================
            try:
                idxs = rng.choice(df.shape[0], min(60, df.shape[0]), replace=False)
                grid = np.linspace(df[feat].min(), df[feat].max(), 20)
                curves = []
                for i in idxs:
                    row = df.iloc[i].copy()
                    curve = []
                    for g in grid:
                        row[feat] = g
                        p = model.predict_proba(
                            preprocessor.transform(pd.DataFrame([row])[FEATURES])
                        )[:, 1][0]
                        curve.append(p)
                    curves.append(curve)
                results[feat]["ICE variance"] = float(np.mean(np.var(curves, axis=0)))
            except Exception:
                results[feat]["ICE variance"] = np.nan

            # ================= LIME =================
            try:
                idxs = rng.choice(X_pre_all.shape[0], min(60, X_pre_all.shape[0]), replace=False)
                weights = []
                for i in idxs:
                    exp = lime_expl.explain_instance(
                        X_pre_all[i], model.predict_proba, num_features=len(FEATURES)
                    )
                    for name, w in exp.as_list():
                        if feat in name:
                            weights.append(abs(w))
                results[feat]["LIME |weight|"] = float(np.mean(weights))
            except Exception:
                results[feat]["LIME |weight|"] = np.nan

            # ================= SHAP =================
            try:
                if vals is not None:
                    idx = FEATURES.index(feat)
                    results[feat]["SHAP mean |value|"] = float(np.mean(np.abs(vals[:, idx])))
                else:
                    results[feat]["SHAP mean |value|"] = np.nan
            except Exception:
                results[feat]["SHAP mean |value|"] = np.nan

            # ================= ANCHOR =================
            try:
                idxs = rng.choice(df.shape[0], min(40, df.shape[0]), replace=False)
                presence = []
                for i in idxs:
                    exp = anchor_expl.explain_instance(
                        df[FEATURES].iloc[i].values, model.predict, threshold=0.95
                    )
                    rule = " ".join(exp.names()).lower()
                    presence.append(1.0 if feat.lower() in rule else 0.0)
                results[feat]["Anchor presence"] = float(np.mean(presence))
            except Exception:
                results[feat]["Anchor presence"] = np.nan

            # ================= ALE =================
            try:
                values = df[feat].values
                bins = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
                ale_vals = []
                for i in range(len(bins) - 1):
                    low, high = bins[i], bins[i + 1]
                    mask = (values >= low) & (values < high)
                    if mask.sum() == 0:
                        continue
                    X_low = df.loc[mask, FEATURES].copy()
                    X_high = X_low.copy()
                    X_low[feat] = low
                    X_high[feat] = high
                    p_low = model.predict_proba(preprocessor.transform(X_low))[:, 1]
                    p_high = model.predict_proba(preprocessor.transform(X_high))[:, 1]
                    ale_vals.append(np.mean(p_high - p_low))
                results[feat]["ALE mean |effect|"] = float(np.mean(np.abs(ale_vals)))
            except Exception:
                results[feat]["ALE mean |effect|"] = np.nan

        # ================= RESULTS TABLE =================
        df_feat = pd.DataFrame(results).T
        st.write("📊 Raw Feature Influence Metrics")
        st.dataframe(df_feat.style.format("{:.4f}"))

        # ================= NORMALIZATION =================
        norm_df = (df_feat - df_feat.min()) / (df_feat.max() - df_feat.min())
        norm_df["Combined Score"] = norm_df.mean(axis=1)

        st.write("📈 Normalized Metrics & Combined Influence Score")
        st.dataframe(norm_df.style.format("{:.4f}"))

        # ================= BAR CHART =================
        fig, ax = plt.subplots(figsize=(7, 4))
        norm_df.sort_values("Combined Score").plot(
            y="Combined Score", kind="barh", ax=ax, legend=False
        )
        ax.set_title("Combined Feature Influence Score (All XAI Methods)")
        st.pyplot(fig)

        st.info("""
        **Interpretation**
        - Higher combined score → feature is consistently important across XAI methods
        - ALE improves reliability when features are correlated
        - Agreement across SHAP + ALE + PDP = strongest evidence
        """)

    # ---------------- TAB: ALE (Accumulated Local Effects) ----------------
    with tab7:
        st.subheader("🟠 ALE (Accumulated Local Effects)")

        st.markdown("""
        **ALE explains how a feature affects PCOS prediction on average,  
        while handling correlated features better than PDP.**
        """)

        feature_ale = st.selectbox(
            "Select feature for ALE",
            FEATURES,
            index=FEATURES.index("amhng_ml")
        )

        try:
            X_raw = df[FEATURES].copy()
            values = X_raw[feature_ale].values

            # ALE configuration
            N_BINS = 10
            bins = np.quantile(values, np.linspace(0, 1, N_BINS + 1))
            bins = np.unique(bins)

            ale_effects = []
            bin_centers = []

            # ---- Compute ALE ----
            for i in range(len(bins) - 1):
                low, high = bins[i], bins[i + 1]

                mask = (values >= low) & (values < high)
                if mask.sum() == 0:
                    continue

                X_low = X_raw.loc[mask].copy()
                X_high = X_low.copy()

                X_low[feature_ale] = low
                X_high[feature_ale] = high

                pred_low = model.predict_proba(
                    preprocessor.transform(X_low)
                )[:, 1]

                pred_high = model.predict_proba(
                    preprocessor.transform(X_high)
                )[:, 1]

                local_effect = np.mean(pred_high - pred_low)

                ale_effects.append(local_effect)
                bin_centers.append((low + high) / 2)

            # Accumulate and center ALE
            ale_effects = np.array(ale_effects)
            ale_cum = np.cumsum(ale_effects)
            ale_cum -= np.mean(ale_cum)

            # ---- Plot ----
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(bin_centers, ale_cum, marker="o", linewidth=2)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel(feature_ale)
            ax.set_ylabel("ALE effect on PCOS probability")
            ax.set_title(f"ALE Plot for {feature_ale}")
            ax.grid(True)

            st.pyplot(fig)

            # ---- Explanation ----
            st.info(
                f"""
                **Interpretation:**  
                - Values above zero increase PCOS probability  
                - Values below zero decrease PCOS probability  
                - ALE accounts for interactions with other hormones
                """
            )

        except Exception as e:
            st.error(f"ALE computation failed: {e}")

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
