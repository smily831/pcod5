# pcos_xai_dashboard_full.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
import shap
from lime.lime_tabular import LimeTabularExplainer
from anchor import anchor_tabular
import warnings

warnings.filterwarnings("ignore")

# ---------- MODEL SELECTION ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Model Selection")

MODEL_MAP = {
    "Random Forest": "model_RandomForestClassifier.joblib",
    "Logistic Regression": "model_LogisticRegression.joblib",
}

# Filter to only show models that exist
existing_models = {k: v for k, v in MODEL_MAP.items() if os.path.exists(v)}
if not existing_models:
    st.error("No model files found. Please run 'train_master.py' first.")
    st.stop()

model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(existing_models.keys()),
    index=0
)

MODEL_FILE = existing_models[model_choice]


# ---------- IMPORTANT helper (must match pipeline) ----------
def clip_nonnegative(X):
    import numpy as _np
    X = _np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X


# ---------- small SHAP helper ----------
def safe_shap_values_and_array(explainer, X_array, prefer_class=1, use_check_additivity=True):
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

FEATURES = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
TARGET = "pcos_y_n"


# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts(model_file):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found")

    preprocessor = joblib.load(PIPELINE_FILE)
    model = joblib.load(model_file)

    # FIX 1: Fill NaNs immediately to prevent Anchor crash
    df = None
    if os.path.exists(CLEANED_CSV):
        df = pd.read_csv(CLEANED_CSV)
        for col in FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    return preprocessor, model, df


try:
    preprocessor, model, df = load_artifacts(MODEL_FILE)
except Exception as e:
    st.error(f"Error loading pipeline/model: {e}")
    st.stop()

# ---------- Global feature selection (shared across tabs) ----------
if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = FEATURES[2]

st.sidebar.header("Feature Selection")
st.session_state.selected_feature = st.sidebar.selectbox(
    "Select feature to analyze (global)",
    FEATURES,
    index=FEATURES.index(st.session_state.selected_feature)
)

# ---- New: choose whether to explain a manual input or a dataset patient ----
st.sidebar.markdown("---")
st.sidebar.subheader("Input patient source")

if df is not None:
    sample_source = st.sidebar.radio("Explain:", ("Manual input", "Select from dataset"))
else:
    sample_source = "Manual input"

defaults = df[FEATURES].median().to_dict() if df is not None else {FEATURES[0]: 20.0, FEATURES[1]: 1.99,
                                                                   FEATURES[2]: 3.5}

selected_dataset_index = None
if sample_source == "Select from dataset":
    id_col = None
    if df is not None:
        for candidate in ["id", "patient_id", "ID", "PatientID"]:
            if candidate in df.columns:
                id_col = candidate
                break

    if df is not None:
        if id_col is not None:
            idx_choice = st.sidebar.selectbox("Choose patient (index — id)", options=list(df.index),
                                              format_func=lambda i: f"{i} — {df.loc[i, id_col]}")
            selected_dataset_index = int(idx_choice)
        else:
            idx_choice = st.sidebar.selectbox("Choose patient (index)", options=list(df.index))
            selected_dataset_index = int(idx_choice)

    if selected_dataset_index is not None:
        sel_row = df.loc[selected_dataset_index, FEATURES]
        st.sidebar.markdown("**Selected patient values:**")
        st.sidebar.write(sel_row.to_frame(name="value"))

        if st.sidebar.button("Use these values (load to manual inputs)"):
            st.session_state["_loaded_beta1"] = float(sel_row[FEATURES[0]])
            st.session_state["_loaded_beta2"] = float(sel_row[FEATURES[1]])
            st.session_state["_loaded_amh"] = float(sel_row[FEATURES[2]])

# ---- Manual input controls ----
st.sidebar.markdown("---")
st.sidebar.subheader("Input patient features (manual)")


def _get_default_feat(feat):
    key_map = {FEATURES[0]: "_loaded_beta1", FEATURES[1]: "_loaded_beta2", FEATURES[2]: "_loaded_amh"}
    if key_map[feat] in st.session_state:
        return float(st.session_state[key_map[feat]])
    return float(defaults.get(feat, 0.0))


beta1 = st.sidebar.number_input("I β-hCG (mIU/mL)", value=_get_default_feat(FEATURES[0]))
beta2 = st.sidebar.number_input("II β-hCG (mIU/mL)", value=_get_default_feat(FEATURES[1]))
amh = st.sidebar.number_input("AMH (ng/mL)", value=_get_default_feat(FEATURES[2]))

if sample_source == "Select from dataset" and df is not None and selected_dataset_index is not None:
    sample_df = df.loc[[selected_dataset_index]][FEATURES].copy().reset_index(drop=True)
else:
    sample_df = pd.DataFrame([{FEATURES[0]: beta1, FEATURES[1]: beta2, FEATURES[2]: amh}])[FEATURES]

st.sidebar.markdown(f"**Active sample source:** `{sample_source}`" + (
    f" — index {selected_dataset_index}" if selected_dataset_index is not None else ""))

st.title("PCOS Prediction using XAI ")
st.markdown("Global feature selector is shared across tabs. Selected: **`%s`**" % st.session_state.selected_feature)

# ---------- Prediction ----------
if st.sidebar.button("Predict PCOS"):
    try:
        X_trans = preprocessor.transform(sample_df)
        prob = float(model.predict_proba(X_trans)[:, 1][0])
        pred = int(model.predict(X_trans)[0])
    except Exception as e:
        st.error(f"Error during preprocessing/prediction: {e}")
        st.stop()

    col1, col2 = st.columns([2, 3])
    with col1:
        st.metric("Predicted Label", "PCOS (1)" if pred == 1 else "No PCOS (0)")
        st.write(f"Probability (PCOS=1): **{prob:.4f}**")
        st.progress(min(max(prob, 0.0), 1.0))
    with col2:
        st.subheader("Input values")
        st.table(sample_df.T.rename(columns={0: "value"}))

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📈 PDP + ICE", "🟢 LIME", "🔵 SHAP", "🟣 ANCHOR", "🟠 ALE", "📊 XAI Comparison", "🎯 Feature Influence"]
    )

    # ---------------- TAB 1: PDP + ICE ----------------
    with tab1:
        st.subheader("PDP + ICE (global & individual effects)")
        if df is None:
            st.warning("Cleaned CSV required for PDP/ICE.")
        else:
            X_orig = df[FEATURES]
            try:
                X_pre = preprocessor.transform(X_orig)
            except Exception as e:
                st.error(f"Could not transform data for PDP: {e}")
                X_pre = None

            pdp_feature = st.session_state.selected_feature
            show_ice = st.checkbox("Show ICE curves", value=True)
            grid_resolution = st.slider("Grid resolution", 20, 200, 80, 10)
            kind = "both" if show_ice else "average"

            if X_pre is not None:
                try:
                    feat_idx = FEATURES.index(pdp_feature)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    PartialDependenceDisplay.from_estimator(
                        estimator=model, X=X_pre, features=[feat_idx],
                        feature_names=FEATURES,  # Added feature_names
                        kind=kind, grid_resolution=grid_resolution, ax=ax
                    )
                    ax.set_title(f"PDP + ICE for {pdp_feature}")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"PDP compute error: {e}")

            st.markdown("**2D PDP (interaction)**")
            try:
                # FIX 2: Let sklearn create figure to avoid "Axis already used" error
                idx_a = FEATURES.index(pdp_feature)
                idx_b = 0 if idx_a != 0 else 1

                disp = PartialDependenceDisplay.from_estimator(
                    estimator=model, X=X_pre, features=[(idx_a, idx_b)],
                    feature_names=FEATURES,  # Added feature_names
                    kind="average", grid_resolution=30
                )
                disp.figure_.set_size_inches(6, 5)
                disp.axes_[0][0].set_title(f"2D PDP: {FEATURES[idx_a]} vs {FEATURES[idx_b]}")
                st.pyplot(disp.figure_)
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
                class_names=["No PCOS", "PCOS"],
                mode="classification"
            )
            sample_pre = preprocessor.transform(sample_df)[0]
            exp = lime_explainer.explain_instance(sample_pre, model.predict_proba, num_features=len(FEATURES))
            st.write("Top local contributions (LIME):")
            st.write(exp.as_list())
            st.pyplot(exp.as_pyplot_figure())
        except Exception as e:
            st.error(f"LIME failed: {e}")

    # ---------------- TAB 3: SHAP ----------------
    with tab3:
        st.subheader("SHAP (global + local)")
        try:
            X_train_pre = preprocessor.transform(df[FEATURES])
            sample_pre = preprocessor.transform(sample_df)

            try:
                masker = shap.maskers.Independent(X_train_pre, max_samples=300)
            except Exception:
                masker = None

            explainer = None
            try:
                if masker is not None:
                    explainer = shap.Explainer(model, masker=masker, model_output="probability")
                else:
                    explainer = shap.Explainer(model, model_output="probability")
            except Exception:
                try:
                    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
                except Exception:
                    raise

            n_sample = min(300, X_train_pre.shape[0])
            idxs = np.random.RandomState(42).choice(X_train_pre.shape[0], n_sample, replace=False)
            X_sample = X_train_pre[idxs]

            vals_global, expl_obj_global = safe_shap_values_and_array(explainer, X_sample, prefer_class=1,
                                                                      use_check_additivity=True)

            st.write("### 🌍 Global SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(7, 4))
            shap.summary_plot(vals_global, X_sample, feature_names=FEATURES, show=False)
            st.pyplot(fig)

            st.write("### 🔍 Local SHAP Waterfall")
            vals_local, expl_obj_local = safe_shap_values_and_array(explainer, sample_pre, prefer_class=1,
                                                                    use_check_additivity=True)
            v = vals_local.ravel()

            base_val = None
            if expl_obj_local is not None and hasattr(expl_obj_local, "base_values"):
                bv = np.array(expl_obj_local.base_values)
                try:
                    if bv.ndim == 2:
                        base_val = float(bv[0, 1]) if bv.shape[1] > 1 else float(bv[0, 0])
                    else:
                        base_val = float(bv.ravel()[-1])
                except Exception:
                    base_val = float(bv.ravel()[0])
            else:
                try:
                    base_val = float(model.predict_proba(sample_pre)[:, 1][0])
                except Exception:
                    base_val = 0.0

            expl_for_waterfall = shap.Explanation(
                values=v,
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
                axb.set_title("Local SHAP contributions")
                st.pyplot(figb)

        except Exception as e:
            st.error(f"SHAP failed: {e}")

    # ---------------- TAB 4: ANCHOR ----------------
    with tab4:
        st.subheader("ANCHOR (rule-based explanations)")

        try:
            # Initialize Anchor explainer
            anchor_explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=["No PCOS", "PCOS"],
                feature_names=FEATURES,
                train_data=df[FEATURES].values
            )

            # Generate explanation
            exp = anchor_explainer.explain_instance(
                sample_df.values[0],
                model.predict,
                threshold=0.95,
                delta=0.1
            )

            # Extract rule, precision, coverage
            rule = " AND ".join(exp.names()) if hasattr(exp, "names") else str(exp)
            precision_val = exp.precision() if callable(getattr(exp, "precision", None)) else getattr(exp, "precision",
                                                                                                      np.nan)
            coverage_val = exp.coverage() if callable(getattr(exp, "coverage", None)) else getattr(exp, "coverage",
                                                                                                   np.nan)

            st.write("Anchor rule:")
            st.info(rule)
            st.write(f"Precision: {precision_val:.3f} | Coverage: {coverage_val:.3f}")

            # ------------------------------------------
            # Visualization of Anchor rule effect
            # ------------------------------------------
            import re
            import seaborn as sns
            import matplotlib.pyplot as plt

            # Get rule text safely
            try:
                rule_text = " AND ".join(exp.names()) if hasattr(exp, "names") else str(exp)
            except Exception:
                rule_text = str(exp)

            # Extract threshold from rule (for i___beta_hcgmiu_ml)
            thresh = None
            match = re.search(r"i___beta_hcgmiu_ml\s*(?:<=|<|>)\s*([0-9.+-eE]+)", rule_text)
            if match:
                try:
                    thresh = float(match.group(1))
                except Exception:
                    thresh = None

            # Fallback threshold if extraction fails
            if thresh is None:
                thresh = 1.99

            # Prepare full dataset predictions
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

            # Compute mask, coverage, precision
            mask = X_all["i___beta_hcgmiu_ml"] <= thresh
            coverage_calc = mask.mean()

            if mask.sum() > 0:
                majority_class = X_all.loc[mask, "_pred"].mode().iat[0]
                precision_calc = (X_all.loc[mask, "_pred"] == majority_class).mean()
            else:
                precision_calc = np.nan

            # ------------------------------------------
            # Plotting
            # ------------------------------------------
            fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)

            # Panel 1: Histogram
            ax = axes[0]
            sns.histplot(
                data=X_all,
                x="i___beta_hcgmiu_ml",
                hue="_pred",
                bins=30,
                multiple="layer",
                alpha=0.6,
                ax=ax
            )
            ax.axvline(thresh, color="black", linestyle="--", linewidth=2,
                       label=f"Anchor threshold = {thresh}")
            ax.set_title("Distribution of I β-hCG by model predicted class")
            ax.set_xlabel("i___beta_hcgmiu_ml")
            ax.legend()

            # Panel 2: Scatter plot
            ax = axes[1]
            ax.scatter(X_all["i___beta_hcgmiu_ml"], X_all["_prob"], alpha=0.6, s=18)
            ax.axvline(thresh, color="black", linestyle="--", linewidth=2)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title("Predicted PCOS probability vs I β-hCG")
            ax.set_xlabel("i___beta_hcgmiu_ml")
            ax.set_ylabel("Predicted probability (PCOS=1)")

            # Panel 3: Precision & Coverage
            ax = axes[2]
            metrics = {
                "Coverage (%)": coverage_calc * 100,
                "Precision (%)": precision_calc * 100
            }
            names = list(metrics.keys())
            values = list(metrics.values())

            ax.barh(names, values, color=["#2a9d8f", "#e76f51"])
            for i, v in enumerate(values):
                ax.text(v + 1, i, f"{v:.1f}%", va="center")

            ax.set_xlim(0, 105)
            ax.set_title("Anchor Rule Metrics")
            ax.set_xlabel("Percentage")

            st.pyplot(fig)

            # ------------------------------------------
            # Example rows
            # ------------------------------------------
            st.write(f"Examples (first 10) where rule holds (i___beta_hcgmiu_ml ≤ {thresh:.2f}):")
            st.dataframe(X_all.loc[mask, FEATURES + ["_pred", "_prob"]].head(10))

            st.write("Examples (first 10) where rule does NOT hold:")
            st.dataframe(X_all.loc[~mask, FEATURES + ["_pred", "_prob"]].head(10))

        except Exception as e:
            st.error(f"Anchor failed: {e}")

    # ---------------- TAB 6: XAI COMPARISON ----------------
    with tab6:
        st.subheader("📊 Quantitative Performance of XAI Techniques")

        if df is None:
            st.warning("Dataset required for XAI comparison.")
            st.stop()

        X_raw = df[FEATURES]
        X_pre = preprocessor.transform(X_raw)
        rng = np.random.RandomState(42)

        idxs = rng.choice(len(X_raw), min(60, len(X_raw)), replace=False)

        # ================= LIME =================
        lime_expl = LimeTabularExplainer(
            training_data=np.array(X_pre),
            feature_names=FEATURES,
            class_names=["No PCOS", "PCOS"],
            mode="classification"
        )

        lime_scores = []
        for i in idxs:
            try:
                exp = lime_expl.explain_instance(
                    X_pre[i],
                    model.predict_proba,
                    num_features=len(FEATURES)
                )
                lime_scores.append(exp.score)
            except:
                lime_scores.append(np.nan)

        lime_fidelity = np.nanmean(lime_scores)

        # ================= SHAP =================
        try:
            shap_expl = shap.TreeExplainer(model)
            shap_vals = shap_expl(X_pre[idxs])

            vals = shap_vals.values
            if vals.ndim == 3:
                vals = vals[:, 1, :]

            base = shap_expl.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = base[1]

            preds = model.predict_proba(X_pre[idxs])[:, 1]
            preds_shap = vals.sum(axis=1) + base
            shap_error = np.mean(np.abs(preds - preds_shap))
        except:
            shap_error = np.nan

        # ================= ANCHOR =================
        anchor_expl = anchor_tabular.AnchorTabularExplainer(
            class_names=["No PCOS", "PCOS"],
            feature_names=FEATURES,
            train_data=X_raw.values
        )

        precisions = []
        for i in idxs[:25]:
            try:
                exp = anchor_expl.explain_instance(
                    X_raw.iloc[i].values,
                    model.predict,
                    threshold=0.95
                )
                precisions.append(exp.precision())
            except:
                precisions.append(np.nan)

        anchor_precision = np.nanmean(precisions)

        # ================= PDP =================
        pdp_slopes = []
        for feat in FEATURES:
            grid = np.linspace(X_raw[feat].min(), X_raw[feat].max(), 20)
            preds = []
            for g in grid:
                X_tmp = X_raw.copy()
                X_tmp[feat] = g
                preds.append(model.predict_proba(preprocessor.transform(X_tmp))[:, 1].mean())
            slope = np.abs(np.gradient(preds)).mean()
            pdp_slopes.append(slope)

        pdp_effect = np.mean(pdp_slopes)

        # ================= ICE =================
        ice_vars = []
        for feat in FEATURES:
            curves = []
            for i in idxs[:30]:
                row = X_raw.iloc[i].copy()
                curve = []
                for g in grid:
                    row[feat] = g
                    curve.append(
                        model.predict_proba(
                            preprocessor.transform(pd.DataFrame([row])[FEATURES])
                        )[:, 1][0]
                    )
                curves.append(curve)
            ice_vars.append(np.var(curves))

        ice_variance = np.mean(ice_vars)

        # ================= ALE =================
        ale_effects = []
        feat = st.session_state.selected_feature
        values = X_raw[feat].values
        bins = np.unique(np.quantile(values, np.linspace(0, 1, 10)))

        for i in range(len(bins) - 1):
            mask = (values >= bins[i]) & (values < bins[i + 1])
            if mask.sum() == 0:
                continue
            X_low = X_raw.loc[mask].copy()
            X_high = X_low.copy()
            X_low[feat] = bins[i]
            X_high[feat] = bins[i + 1]
            p_low = model.predict_proba(preprocessor.transform(X_low))[:, 1]
            p_high = model.predict_proba(preprocessor.transform(X_high))[:, 1]
            ale_effects.append(np.mean(p_high - p_low))

        ale_effect = np.mean(np.abs(ale_effects)) if ale_effects else 0.0

        # ================= RESULT TABLE =================
        comp_df = pd.DataFrame({
            "XAI Method": ["LIME", "SHAP", "ANCHOR", "PDP", "ICE", "ALE"],
            "Metric Used": [
                "Fidelity (R²)",
                "Additivity Error",
                "Rule Precision",
                "Global Slope",
                "Variance",
                "Mean Effect"
            ],
            "Score": [
                lime_fidelity,
                shap_error,
                anchor_precision,
                pdp_effect,
                ice_variance,
                ale_effect
            ]
        })

        st.dataframe(comp_df.style.format({"Score": "{:.4f}"}))

        # ================= BAR CHART =================
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(comp_df["XAI Method"], comp_df["Score"])
        ax.set_title("XAI Technique Comparison")
        st.pyplot(fig)

        # ================= INTERPRETATION =================
        st.info("""
        **How to interpret this comparison:**
        - Higher LIME fidelity → better local explanation accuracy  
        - Lower SHAP error → more consistent explanations  
        - Higher ANCHOR precision → more reliable decision rules  
        - Higher PDP/ALE → stronger global feature influence  
        - Lower ICE variance → more stable behavior across patients  

        No single XAI method is best for all purposes.  
        Combining multiple XAI techniques provides the most trustworthy interpretation.
        """)

    # ---------------- TAB 7: Feature Influence ----------------
    with tab7:
        st.subheader("🎯 Feature Influence Dashboard")
        st.write("Aggregated influence scores across multiple XAI methods.")

        if df is None:
            st.warning("Dataset required.")
            st.stop()

        X_raw = df[FEATURES].copy()
        X_pre = preprocessor.transform(X_raw)
        rng = np.random.RandomState(42)

        results = {f: {} for f in FEATURES}

        # ========== SHAP ==========
        try:
            shap_expl = shap.TreeExplainer(model)
            shap_vals = shap_expl(X_pre)
            vals = shap_vals.values
            if vals.ndim == 3:
                vals = vals[:, 1, :]
            for i, f in enumerate(FEATURES):
                results[f]["SHAP"] = np.mean(np.abs(vals[:, i]))
        except:
            for f in FEATURES:
                results[f]["SHAP"] = np.nan

        # ========== LIME ==========
        lime_expl = LimeTabularExplainer(
            training_data=np.array(X_pre),
            feature_names=FEATURES,
            class_names=["No PCOS", "PCOS"],
            mode="classification"
        )

        for f in FEATURES:
            weights = []
            for i in rng.choice(len(X_pre), min(40, len(X_pre)), replace=False):
                try:
                    exp = lime_expl.explain_instance(X_pre[i], model.predict_proba)
                    for name, w in exp.as_list():
                        if f in name:
                            weights.append(abs(w))
                except:
                    pass
            results[f]["LIME"] = np.mean(weights) if weights else np.nan

        # ========== PDP ==========
        for f in FEATURES:
            grid = np.linspace(X_raw[f].min(), X_raw[f].max(), 20)
            preds = []
            for g in grid:
                X_tmp = X_raw.copy()
                X_tmp[f] = g
                preds.append(model.predict_proba(preprocessor.transform(X_tmp))[:, 1].mean())
            results[f]["PDP"] = np.mean(np.abs(np.gradient(preds)))

        # ========== ICE ==========
        for f in FEATURES:
            curves = []
            for i in rng.choice(len(X_raw), min(30, len(X_raw)), replace=False):
                row = X_raw.iloc[i].copy()
                curve = []
                for g in grid:
                    row[f] = g
                    curve.append(
                        model.predict_proba(
                            preprocessor.transform(pd.DataFrame([row])[FEATURES])
                        )[:, 1][0]
                    )
                curves.append(curve)
            results[f]["ICE"] = np.var(curves)

        # ========== ANCHOR ==========
        anchor_expl = anchor_tabular.AnchorTabularExplainer(
            class_names=["No PCOS", "PCOS"],
            feature_names=FEATURES,
            train_data=X_raw.values
        )

        for f in FEATURES:
            presence = []
            for i in rng.choice(len(X_raw), min(25, len(X_raw)), replace=False):
                try:
                    exp = anchor_expl.explain_instance(X_raw.iloc[i].values, model.predict)
                    rule = " ".join(exp.names()).lower()
                    presence.append(1 if f.lower() in rule else 0)
                except:
                    presence.append(0)
            results[f]["ANCHOR"] = np.mean(presence)

        # ========== ALE ==========
        for f in FEATURES:
            values = X_raw[f].values
            bins = np.unique(np.quantile(values, np.linspace(0, 1, 10)))
            effects = []
            for i in range(len(bins) - 1):
                mask = (values >= bins[i]) & (values < bins[i + 1])
                if mask.sum() == 0:
                    continue
                X_low = X_raw.loc[mask].copy()
                X_high = X_low.copy()
                X_low[f] = bins[i]
                X_high[f] = bins[i + 1]
                p_low = model.predict_proba(preprocessor.transform(X_low))[:, 1]
                p_high = model.predict_proba(preprocessor.transform(X_high))[:, 1]
                effects.append(np.mean(p_high - p_low))
            results[f]["ALE"] = np.mean(np.abs(effects)) if effects else np.nan

        # ========== TABLE ==========
        df_feat = pd.DataFrame(results).T
        st.write("### Raw Feature Influence Scores")
        st.dataframe(df_feat.style.format("{:.4f}"))

        # ========== NORMALIZATION ==========
        norm_df = (df_feat - df_feat.min()) / (df_feat.max() - df_feat.min())
        norm_df["Combined Score"] = norm_df.mean(axis=1)

        st.write("### Normalized & Combined Influence Score")
        st.dataframe(norm_df.style.format("{:.4f}"))

        # ========== BAR CHART ==========
        fig, ax = plt.subplots(figsize=(7, 4))
        norm_df.sort_values("Combined Score").plot(
            y="Combined Score", kind="barh", ax=ax, legend=False
        )
        ax.set_title("Overall Feature Importance (Aggregated XAI)")
        st.pyplot(fig)

        st.info("""
        **How to read this dashboard:**
        - Higher combined score → feature is consistently important across XAI methods  
        - Agreement across SHAP + PDP + ALE = strong clinical relevance  
        - ICE shows patient-level variability  
        - ANCHOR confirms rule-level importance  
        """)

    # ---------------- TAB 5: ALE ----------------
    with tab5:
        st.subheader("🟠 ALE (Accumulated Local Effects)")
        feat_ale = st.session_state.selected_feature

        if df is not None:
            try:
                X_raw = df[FEATURES].copy()
                values = X_raw[feat_ale].values
                hist_bins = min(20, len(np.unique(values)) - 1)
                bins = np.linspace(values.min(), values.max(), hist_bins + 1)

                ale_effects = []
                bin_centers = []

                for i in range(len(bins) - 1):
                    low, high = bins[i], bins[i + 1]
                    mask = (values >= low) & (values < high)
                    if i == len(bins) - 2: mask = (values >= low) & (values <= high)

                    if mask.sum() > 0:
                        X_low = X_raw.loc[mask].copy();
                        X_high = X_low.copy()
                        X_low[feat_ale] = low;
                        X_high[feat_ale] = high
                        p_low = model.predict_proba(preprocessor.transform(X_low))[:, 1]
                        p_high = model.predict_proba(preprocessor.transform(X_high))[:, 1]
                        ale_effects.append(np.mean(p_high - p_low))
                        bin_centers.append((low + high) / 2)

                if ale_effects:
                    ale_cum = np.cumsum(ale_effects)
                    ale_cum -= np.mean(ale_cum)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(bin_centers, ale_cum, marker="o")
                    ax.axhline(0, color="k", linestyle="--")
                    ax.set_title(f"ALE for {feat_ale}")
                    st.pyplot(fig)
                else:
                    st.warning("Not enough variance for ALE.")
            except Exception as e:
                st.error(f"ALE Error: {e}")

# Footer
st.markdown("---")
st.caption("PCOS XAI Dashboard | Powered by Streamlit, SHAP, LIME, Anchor & Scikit-Learn")