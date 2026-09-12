import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import shap

# -----------------------------------------------------------------------
# Globals

SMALLPLOT_WIDTH = 500
TEXT_WIDTH = 700
model = None
prediction = None

# Initialize persistent session state tracking for time metrics
if "total_time_saved_minutes" not in st.session_state:
    st.session_state.total_time_saved_minutes = 0.0

# -----------------------------------------------------------------------
# Helpful defs

def wide_centered_layout():
    with st.container(horizontal_alignment="center"):
        return st.container(
            width=2 * SMALLPLOT_WIDTH + 16, horizontal_alignment="center"
        )
    
@st.cache_resource
def load_model_file(filepath: str):
    """
    Loads a unified model payload file. Explicitly extracts the scikit-learn 
    pipeline object away from its surrounding metadata wrapper dictionary.
    """
    ext = filepath.split(".")[-1]
    
    if ext == "joblib":
        payload = joblib.load(filepath)
    elif ext == "pkl":
        with open(filepath, 'rb') as file:
            payload = pickle.load(file)
    else:
        raise ValueError(f"Unable to handle file extension: {ext}")

    # Extract the underlying model if wrapped in our tuner's artifact payload dictionary
    if isinstance(payload, dict) and "model" in payload:
        model_obj = payload["model"]
        metadata = {
            "score": payload.get("best_score", -1),
            "optimized_threshold": payload.get("best_threshold", 0.5),
            "metric_optimized": payload.get("metric_optimized", "unknown"),
            "classes": payload.get("classes", ["no", "yes"]),
            "info": f"Production pipeline optimized for operational {str(payload.get('metric_optimized', '')).upper()}"
        }
    else:
        # Fallback for standard standalone files
        model_obj = payload
        try:
            meta_path = filepath.replace(f".{ext}", f"_metadata.{ext}")
            if ext == "joblib":
                metadata = joblib.load(meta_path)
            else:
                with open(meta_path, 'rb') as file:
                    metadata = pickle.load(file)
        except Exception:
            metadata = {"score": -1, "optimized_threshold": 0.5, "info": "No metadata found on selected model asset"}

    return model_obj, metadata

@st.cache_resource
def get_paired_base_models(model_folder: str):
    """
    Scans the folder and returns base names of models that have 
    both an '_efficient' and a '_max_sales' version.
    """
    path = Path(model_folder)
    if not path.exists():
        return []
    
    all_files = [f.name for f in path.iterdir() if f.is_file() and "_metadata" not in f.name]
    
    base_names = set()
    for f in all_files:
        name_we = f.rsplit(".", 1)[0]
        if name_we.endswith("_efficient"):
            base_names.add(name_we.rsplit("_efficient", 1)[0])
        elif name_we.endswith("_max_sales"):
            base_names.add(name_we.rsplit("_max_sales", 1)[0])
            
    valid_bases = []
    for base in sorted(base_names):
        eff_exists = any(f.startswith(base + "_efficient") for f in all_files)
        max_exists = any(f.startswith(base + "_max_sales") for f in all_files)
        if eff_exists and max_exists:
            valid_bases.append(base)
            
    return valid_bases

@st.cache_data
def _calculate_shap_values(_threshold_model, dataInput):
    """
    Recursively unwraps TunedThresholdClassifierCV and Scikit-Learn Pipelines 
    to extract the raw fitted Random Forest model and pre-transform data inputs.
    """
    current_model = _threshold_model
    transformed_data = dataInput.copy()

    # 1. Step out of the TunedThresholdClassifierCV wrapper
    if hasattr(current_model, "estimator_"):
        current_model = current_model.estimator_
    elif hasattr(current_model, "best_estimator_"):
        current_model = current_model.best_estimator_

    # 2. Handle a Scikit-Learn Pipeline wrapper step-by-step
    if current_model.__class__.__name__ == "Pipeline":
        # Process data through encoders/scalers up to but excluding the final random forest model
        for name, step in current_model.steps[:-1]:
            if hasattr(step, "transform"):
                transformed_data = step.transform(transformed_data)
        actual_tree_model = current_model.steps[-1][1]
    else:
        actual_tree_model = current_model

    # 3. Handle DataFrame structures safely for the underlying tree model matrix expectations
    if isinstance(transformed_data, pd.DataFrame):
        feature_names = transformed_data.columns.tolist()
        feature_matrix = transformed_data.values
    else:
        feature_names = [f"Feature {i}" for i in range(transformed_data.shape[1])] if hasattr(transformed_data, 'shape') else None
        feature_matrix = transformed_data

    # 4. Generate the SHAP values using the isolated raw Random Forest
    explainer = shap.TreeExplainer(actual_tree_model)
    shap_explanation = explainer(feature_matrix)
    
    # Re-apply structural feature labels back to the SHAP metadata canvas
    if feature_names:
        shap_explanation.feature_names = feature_names
    return shap_explanation

def explain_tree_model(treeModel, dataInput, targetClass = None, title = "Key Drivers Behind This Prediction"):
    try:
        shap_explanation = _calculate_shap_values(treeModel, dataInput)
        if len(shap_explanation.shape) == 3:
            single_shap_values = shap_explanation[0, :, 1] if targetClass is None else shap_explanation[0, :, targetClass]
        else:
            single_shap_values = shap_explanation
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(single_shap_values, show=False)
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.info(f"SHAP chart plotting skipped: Base tree explainer not supported for this wrapper structure ({e}).")

def explain_cohort_dependence(treeModel, test_data, feature_x, feature_y):
    """Generates a SHAP dependence plot to show cohort interactions."""
    try:
        shap_explanation = _calculate_shap_values(treeModel, test_data)
        
        if len(shap_explanation.shape) == 3:
            shap_disp = shap_explanation.values[:, :, 1]
            matrix_data = shap_explanation.data
        else:
            shap_disp = shap_explanation.values
            matrix_data = shap_explanation.data

        df_disp = pd.DataFrame(matrix_data, columns=shap_explanation.feature_names)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feature_x, 
            shap_disp, 
            df_disp, 
            interaction_index=feature_y, 
            ax=ax, 
            show=False
        )
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Risk Threshold')
        plt.title(f"Feature Interaction Analysis: {feature_x} vs. {feature_y}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.info(f"Could not compute background cohort analysis maps: {e}")


# -----------------------------------------------------------------------
# Draw app

with wide_centered_layout():
    with st.container(width=TEXT_WIDTH):
        st.title("Predictive Dialer & Operational Optimizer")
        st.caption("AI-powered operational decision logic to maximize bank term deposit conversions and optimize floor efficiency.")
        
        # Operational Running Metrics Summary Cards
        st.markdown("### 📈 Call-Floor Performance Tracking")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Total Representative Time Saved", f"{st.session_state.total_time_saved_minutes:,.1f} mins")
        with m_col2:
            if st.button("🔄 Reset Running Productivity Metrics"):
                st.session_state.total_time_saved_minutes = 0.0
                st.rerun()
        st.caption("ℹ️ *Note on KPI Approximation:* Time savings are derived by calculating the call `duration` profile of entries matching low sensitivity boundaries that are filtered out of live queues.")
        st.write("")

        modelFolder = "src/models/saved_models/"
        available_bases = get_paired_base_models(modelFolder)
        
        if not available_bases:
            st.warning("⚠️ **No paired strategy models found.** Please ensure your model files are saved in the models directory with matching prefix signatures ending in `_efficient` and `_max_sales` (e.g., `RandomForest_efficient.joblib` and `RandomForest_max_sales.joblib`).")
        else:
            with st.expander("🔍 Model Family Selection & Performance Insights", expanded=False):
                baseSelected = st.segmented_control(
                    "**Choose Base AI Model Engine**",
                    available_bases,
                    default=available_bases,
                )
                
                st.markdown("### Business Strategy Routing Override")
                strategy = st.radio(
                    "Select Active Operational Execution Strategy:",
                    ["Call Conversion Efficiency (_efficient)", "Maximum Conversion Scale (_max_sales)"],
                    help="Efficiency Strategy targets high-precision leads to reduce labor overhead. Max Conversion Strategy targets overall volume expansion."
                )
                
                suffix = "_efficient" if "Efficiency" in strategy else "_max_sales"
                
                path_dir = Path(modelFolder)
                target_file = None
                for f in path_dir.iterdir():
                    if f.name.startswith(baseSelected + suffix):
                        target_file = f.name
                        break
                
                if target_file is not None:
                    model, metadata = load_model_file(modelFolder + target_file)
                    st.markdown("---")
                    st.markdown(f"### Selected Strategy Variant: `{target_file}`")
                    st.markdown("#### Model Architecture")
                    st.write(model)
                    st.markdown("#### Performance Metrics & Optional Metadata")
                    st.write(metadata)

        tab1, tab2 = st.tabs(["👤 Target Individual Client", "🎯 Bulk Queue Segmenter"])

        with tab1:
            st.markdown("### Lead Campaign Attributes")
            st.markdown("Adjust the values below based on client demographic and telemetry data.")

            inputs = {}
            inputs["age"] = st.slider("1. Customer Age", min_value=18, max_value=100, value=35, step=1)
            inputs["job"] = st.selectbox("2. Type of Job", ["management", "technician", "blue-collar", "admin.", "services", "retired", "self-employed", "unemployed", "entrepreneur", "housemaid", "student", "unknown"])
            inputs["marital"] = st.selectbox("3. Marital Status", ["married", "single", "divorced"])
            inputs["education"] = st.selectbox("4. Education Level", ["secondary", "tertiary", "primary", "unknown"])
            inputs["default"] = st.selectbox("5. Has Credit in Default?", ["no", "yes"])
            inputs["balance"] = st.number_input("6. Average Yearly Balance (Euros)", value=1500, step=100)
            inputs["housing"] = st.selectbox("7. Has a Housing Loan?", ["no", "yes"])
            inputs["loan"] = st.selectbox("8. Has a Personal Loan?", ["no", "yes"])
            inputs["contact"] = st.selectbox("9. Contact Communication Type", ["cellular", "telephone", "unknown"])
            inputs["day"] = st.slider("10. Last Contact Day of Month", min_value=1, max_value=31, value=15, step=1)
            inputs["month"] = st.selectbox("11. Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], index=4)
            inputs["duration"] = st.number_input("12. Last Contact Duration (Seconds)", min_value=0, value=240, step=10, help="Interaction length proxy. Rushing clients here harms performance.")
            inputs["campaign"] = st.slider("13. Contacts Performed During Campaign", min_value=1, max_value=50, value=1, step=1)
        
            if model is not None:
                default_features = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign"]
                features = metadata.get("features", default_features) if isinstance(metadata, dict) else default_features
                
                predictionInputs = [inputs[f] for f in features if f in inputs]
                predictionInputs = pd.DataFrame([predictionInputs], columns=features)

                st.markdown("---")
                st.subheader("Automated Lead Routing Verdict")

                prob_converting = model.predict_proba(predictionInputs)[:, 1]
                classification_verdict = model.predict(predictionInputs)

                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.metric("Model Conversion Probability", f"{prob_converting[0]:.2%}")
                with v_col2:
                    if classification_verdict[0] == 1 or classification_verdict[0] == 'yes':
                        st.success("🎯 **DIAL LEAD** (Class: Yes)")
                    else:
                        st.error("🛑 **SKIP RECORD** (Class: No)")
                        
                if st.button("Process Entry & Log Action"):
                    if not (classification_verdict[0] == 1 or classification_verdict[0] == 'yes'):
                        saved_mins = inputs["duration"] / 60.0
                        st.session_state.total_time_saved_minutes += saved_mins
                        st.toast(f"Logged! Saved {saved_mins:.2f} representative operational minutes.")
                        st.rerun()
                    else:
                        st.toast("Lead routed to floor queues. No dial tracking modifications applied.")

                st.write("")
                # explain_tree_model(
                #     model, 
                #     predictionInputs, 
                #     targetClass=1, 
                #     title="Telemetry Elements Driving This Route Assignment"
                # )

        with tab2:
            st.subheader("Queue Batch Segmentation Engine")
            st.markdown(
                "Upload a bulk campaign lead list to parse customer records, calculate conversion likelihood metrics, and sort elements by descending transaction potential."
            )
            
            uploaded_file = st.file_uploader(
                "Upload Campaign Target Data (CSV)", 
                type=["csv"],
                help="Upload a CSV containing bank marketing columns matching model training schema parameters."
            )
            
            if uploaded_file is not None and model is not None:
                batch_df = pd.read_csv(uploaded_file)
                default_features = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign"]
                features_list = metadata.get("features", default_features) if isinstance(metadata, dict) else default_features
                
                if not all(col in batch_df.columns for col in features_list):
                    st.error(f"⚠️ **Incompatible CSV Format:** The uploaded file must include the following column headers: `{features_list}`")
                else:
                    eval_df = batch_df[features_list].dropna()
                    
                    st.markdown("### 1. Feature Interaction Explorer")
                    col1, col2 = st.columns(2)
                    with col1:
                        feat_x = st.selectbox("Primary Factor (X-Axis)", features_list, index=11)
                    with col2:
                        feat_y = st.selectbox("Secondary Comparison Factor (Color)", features_list, index=10)
                    
                    # explain_cohort_dependence(model, eval_df, feat_x, feat_y)
                    
                    st.markdown("### 2. Prioritized Outbound Target Queue (Ranked By Conversion Score)")
                    
                    probabilities = model.predict_proba(eval_df)[:, 1]
                    hard_predictions = model.predict(eval_df)
                    
                    ranked_queue = batch_df.copy()
                    ranked_queue["Conversion_Probability"] = probabilities
                    ranked_queue["Routing_Verdict"] = ["DIAL" if p in [1, 'yes'] else "SKIP" for p in hard_predictions]
                    
                    ranked_queue = ranked_queue.sort_values(by="Conversion_Probability", ascending=False)
                    
                    skipped_leads = ranked_queue[ranked_queue["Routing_Verdict"] == "SKIP"]
                    potential_savings = skipped_leads["duration"].sum() / 60.0
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        st.metric("Total Imported Records", len(ranked_queue))
                    with b_col2:
                        st.metric("Potential Time Optimization in List", f"{potential_savings:,.1f} mins")
                        
                    if st.button("📥 Commit Skip List to Floor Running Totals"):
                        st.session_state.total_time_saved_minutes += potential_savings
                        st.success(f"Added {potential_savings:,.1f} minutes to the master call floor efficiency dashboard!")
                        st.rerun()
                    
                    st.dataframe(ranked_queue, use_container_width=True)
                    
                    csv_data = ranked_queue.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Export Ranked Outbound Dialing Queue (CSV)",
                        data=csv_data,
                        file_name="ranked_outbound_dialer_queue.csv",
                        mime="text/csv",
                    )
