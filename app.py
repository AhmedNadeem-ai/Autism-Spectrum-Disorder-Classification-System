import streamlit as st
import pandas as pd
import joblib
import os
import warnings
from collections import Counter

st.set_page_config(
    page_title="ASD Traits Prediction System",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .result-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 160px;
            padding: 24px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(128,128,128,0.2);
            margin-bottom: 1rem;
        }
        .result-card h3 { 
            margin: 12px 0 !important; 
            font-size: 28px !important; 
            font-weight: 700 !important; 
            line-height: 1.2 !important;
            padding: 0 !important;
        }
        .result-card p { 
            margin: 0 !important; 
            padding: 0 !important;
            font-size: 14px; 
            font-weight: 500; 
            opacity: 0.85; 
        }
        .result-card .title { 
            font-size: 12px; 
            font-weight: 600; 
            text-transform: uppercase; 
            letter-spacing: 1px; 
            margin-bottom: 4px !important;
        }
        .status-positive { 
            background-color: rgba(46, 125, 50, 0.08); 
            border-left: 5px solid #2e7d32;
        }
        .status-negative { 
            background-color: rgba(198, 40, 40, 0.08); 
            border-left: 5px solid #c62828;
        }
        .status-neutral { 
            background-color: rgba(128, 128, 128, 0.08); 
            border-left: 5px solid #9e9e9e;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Autism Spectrum Disorder (ASD) Traits Predictor")
st.markdown("""
This clinical decision support tool utilizes trained Machine Learning models 
to evaluate behavioral, screening, and demographic indicators for early ASD trait screening.
""")

MODEL_FILES = {
    "Decision Tree (Top Model)": os.path.join("supervised_models", "decision_tree.pkl"),
    "K-Nearest Neighbors (KNN)": os.path.join("supervised_models", "knn.pkl"),
    "Support Vector Machine (SVM)": os.path.join("supervised_models", "svm.pkl"),
    "Logistic Regression (L1)": os.path.join("supervised_models", "l1_logistic.pkl"),
    "Logistic Regression (L2)": os.path.join("supervised_models", "l2_logistic.pkl"),
    "Logistic Regression (Base)": os.path.join("supervised_models", "logistic.pkl"),
    "Gaussian Naive Bayes": os.path.join("supervised_models", "naive_bayes.pkl"),
    "K-Means (k=2)": os.path.join("unsupervised_models", "kmeans.pkl"),
    "Gaussian Mixture Model": os.path.join("unsupervised_models", "gmm.pkl"),
    "Divisive K-Means": os.path.join("unsupervised_models", "divisive_kmeans.pkl")
}

MODEL_ACCURACY = {
    "Decision Tree (Top Model)": "95.96%",
    "K-Nearest Neighbors (KNN)": "93.69%",
    "Support Vector Machine (SVM)": "93.18%",
    "Logistic Regression (L1)": "91.16%",
    "Logistic Regression (L2)": "88.38%",
    "Logistic Regression (Base)": "88.38%",
    "Gaussian Naive Bayes": "71.46%",
    "K-Means (k=2)": "66.15%",
    "Gaussian Mixture Model": "66.15%",
    "Divisive K-Means": "66.15%"
}


@st.cache_resource
def load_pipeline_components():
    models = {}
    load_errors = {}

    for name, filepath in MODEL_FILES.items():
        try:
            models[name] = joblib.load(filepath)
        except Exception as e:
            models[name] = None
            load_errors[name] = f"{type(e).__name__}: {e}"

    pca, pca_error = None, None
    pca_path = os.path.join("unsupervised_models", "pca.pkl")
    if os.path.exists(pca_path):
        try:
            pca = joblib.load(pca_path)
        except Exception as e:
            pca_error = f"{type(e).__name__}: {e}"
    else:
        pca_error = "pca.pkl missing from unsupervised_models/"

    scaler, scaler_error = None, None
    scaler_path = os.path.join("supervised_models", "scaler.pkl")
    if not os.path.exists(scaler_path) and os.path.exists("scaler.pkl"):
        scaler_path = "scaler.pkl"

    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            scaler_error = f"{type(e).__name__}: {e}"
    else:
        scaler_error = "scaler.pkl missing from directories"

    return models, load_errors, pca, pca_error, scaler, scaler_error


models, load_errors, pca, pca_error, scaler, scaler_error = load_pipeline_components()
available_models = {name: m for name, m in models.items() if m is not None}

with st.expander(
        f"System Status: {len(available_models)} / {len(MODEL_FILES)} models loaded",
        expanded=bool(load_errors) or pca is None or scaler is None
):
    status_rows = []
    for name in MODEL_FILES:
        ok = models[name] is not None
        status_rows.append({
            "Component": name,
            "Status": "Loaded" if ok else "Failed to load",
            "Detail": "" if ok else load_errors.get(name, "Unknown error")
        })

    status_rows.append({
        "Component": "PCA Transformer (For Unsupervised)",
        "Status": "Loaded" if pca is not None else "Missing",
        "Detail": "" if pca is not None else pca_error
    })

    status_rows.append({
        "Component": "Data Scaler (For Normalization)",
        "Status": "Loaded" if scaler is not None else "Missing",
        "Detail": "" if scaler is not None else scaler_error
    })

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

if not available_models:
    st.error("No trained models are currently available. Please check your system configuration.")
    st.stop()

if scaler is None:
    st.warning(
        "Data Scaler is missing. The system is attempting to predict on unscaled inputs, which may impact distance-based models.")

st.markdown("---")

st.header("Patient Information & Clinical Indicators")
st.caption("Fields marked with an asterisk (*) are required for analysis.")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.subheader("Demographics & History")
        age = st.number_input("Age (Years, 1-18) *", min_value=1, max_value=18, value=None)
        sex = st.radio("Sex *", options=["Male", "Female"], index=None, horizontal=True)
        ethnicity = st.selectbox("Ethnicity",
                                 ["Asian", "White-European", "Middle Eastern", "Black", "Hispanic", "Others"],
                                 index=None)
        jaundice = st.radio("Born with Jaundice?", options=["Yes", "No"], index=None, horizontal=True)
        family_asd = st.radio("Family Member with ASD? *", options=["Yes", "No"], index=None, horizontal=True)

with col2:
    with st.container(border=True):
        st.subheader("Clinical Assessments")
        qchat_10 = st.number_input("QCHAT-10 Score (0-10) *", min_value=0, max_value=10, value=None)
        cars_score = st.number_input("Childhood Autism Rating Scale (1-4) *", min_value=1, max_value=4, value=None)
        srs_score = st.number_input("Social Responsiveness Scale (0-10) *", min_value=0, max_value=10, value=None)
        speech_delay = st.radio("Speech / Language Delay? *", options=["Yes", "No"], index=None, horizontal=True)
        learning_disorder = st.radio("Learning Disorder? *", options=["Yes", "No"], index=None, horizontal=True)

with col3:
    with st.container(border=True):
        st.subheader("Co-occurring Conditions")
        genetic_disorder = st.radio("Genetic Disorders? *", options=["Yes", "No"], index=None, horizontal=True)
        dev_delay = st.radio("Global Developmental Delay? *", options=["Yes", "No"], index=None, horizontal=True)
        social_issues = st.radio("Social / Behavioral Issues? *", options=["Yes", "No"], index=None, horizontal=True)
        anxiety = st.radio("Anxiety Disorder? *", options=["Yes", "No"], index=None, horizontal=True)
        depression = st.radio("Depression? *", options=["Yes", "No"], index=None, horizontal=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Autism Spectrum Quotient (AQ-10) *")

    aq_questions = [
        "A1: Does not easily respond to social cues or unspoken meanings",
        "A2: Strongly prefers repetitive routines and is unsettled by changes",
        "A3: Struggles to understand the social intentions behind what people say",
        "A4: Focuses heavily on small details, patterns, or visual elements that others miss",
        "A5: Has an unusual speech rhythm, tone, or intonation",
        "A6: Finds it difficult to understand other people's feelings",
        "A7: Experiences strong sensitivities to everyday sensory inputs (sounds, lights, textures)",
        "A8: Shows intense, restricted focus on specific topics of interest",
        "A9: Struggles with the natural back-and-forth flow of conversation",
        "A10: Finds unexpected changes to plans or schedules very difficult to handle"
    ]

    aq_col1, aq_col2 = st.columns(2, gap="large")
    aq_responses = {}

    for idx, q_text in enumerate(aq_questions):
        target_col = aq_col1 if idx < 5 else aq_col2
        with target_col:
            val = st.radio(q_text, ["Yes", "No"], index=None, key=f"aq_{idx + 1}", horizontal=True)
            aq_responses[f"A{idx + 1}"] = 1 if val == "Yes" else (0 if val == "No" else None)

st.markdown("---")

run = st.button("Run Multi-Model Diagnostic Evaluation", type="primary", use_container_width=True)

if run:
    required_inputs = {
        "Age": age, "Sex": sex, "Family member with ASD": family_asd,
        "QCHAT-10 Score": qchat_10, "CARS Score": cars_score, "SRS Score": srs_score,
        "Speech/Language Delay": speech_delay, "Learning Disorder": learning_disorder,
        "Genetic Disorders": genetic_disorder, "Developmental Delay": dev_delay,
        "Social/Behavioral Issues": social_issues, "Anxiety Disorder": anxiety, "Depression": depression
    }

    missing = [label for label, v in required_inputs.items() if v is None]
    missing += [f"AQ-10 Question {k}" for k, v in aq_responses.items() if v is None]

    if missing:
        st.error("Evaluation cannot proceed. Please complete all required fields indicated by an asterisk (*).")
        st.stop()

    b_map = {"No": 0, "Yes": 1}

    input_data = {
        'A1': aq_responses['A1'],
        'A2': aq_responses['A2'],
        'A3': aq_responses['A3'],
        'A4': aq_responses['A4'],
        'A5': aq_responses['A5'],
        'A6': aq_responses['A6'],
        'A7': aq_responses['A7'],
        'A8': aq_responses['A8'],
        'A9': aq_responses['A9'],
        'A10': aq_responses['A10'],
        'Social_Responsiveness_Scale': srs_score,
        'Age_Years': age,
        'Qchat_10_Score': qchat_10,
        'Speech_Delay': b_map[speech_delay],
        'Learning_Disorder': b_map[learning_disorder],
        'Genetic_Disorders': b_map[genetic_disorder],
        'Depression': b_map[depression],
        'Global_Developmental_Delay': b_map[dev_delay],
        'Social_Behavioural_Issues': b_map[social_issues],
        'Childhood_Autism_Rating_Scale': cars_score,
        'Anxiety_Disorder': b_map[anxiety],
        'Sex_Male': 1 if sex == "Male" else 0,
        'Family_Mem_With_ASD': b_map[family_asd]
    }

    input_df = pd.DataFrame([input_data])
    X_raw = input_df.to_numpy()

    if scaler is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = X_raw

    results = {}
    probabilities = {}
    prediction_errors = {}

    with st.spinner("Processing clinical data across models..."):
        for name, model in available_models.items():
            try:
                expected_features = getattr(model, "n_features_in_", 23)

                if expected_features == 2:
                    if pca is None:
                        raise ValueError("Model requires PCA compression. 'pca.pkl' is missing.")
                    X_for_model = pca.transform(X_scaled)
                else:
                    X_for_model = X_scaled

                pred = model.predict(X_for_model)[0]
                results[name] = "ASD Traits Detected" if pred == 1 else "No ASD Traits Detected"

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_for_model)[0]
                    probabilities[name] = max(probs) * 100
                else:
                    probabilities[name] = None

            except Exception as e:
                prediction_errors[name] = f"{type(e).__name__}: {e}"

    if prediction_errors:
        with st.expander(f"{len(prediction_errors)} model(s) encountered errors during execution", expanded=True):
            for name, err in prediction_errors.items():
                st.write(f"**{name}:** {err}")

    if not results:
        st.error("Diagnostic evaluation failed. No models returned a valid prediction.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Evaluation Results")

    all_predictions = list(results.values())
    consensus_pred, count = Counter(all_predictions).most_common(1)[0]
    total_models = len(all_predictions)

    top_model_name = "Decision Tree (Top Model)"

    res_col1, res_col2, res_col3 = st.columns(3, gap="medium")


    def create_result_card(title, value, subtext, status_reference):
        if "No ASD" in status_reference:
            css_class = "status-positive"
        elif "ASD Traits Detected" in status_reference:
            css_class = "status-negative"
        else:
            css_class = "status-neutral"
        return f'<div class="result-card {css_class}"><p class="title">{title}</p><h3>{value}</h3><p>{subtext}</p></div>'


    with res_col1:
        st.markdown(create_result_card("Primary Assessment", results.get(top_model_name, "N/A"), "Decision Tree Model",
                                       results.get(top_model_name, "")), unsafe_allow_html=True)

    with res_col2:
        conf = probabilities.get(top_model_name)
        st.markdown(create_result_card("Confidence Interval", f"{conf:.1f}%" if conf else "N/A", "Probability Score",
                                       results.get(top_model_name, "")), unsafe_allow_html=True)

    with res_col3:
        st.markdown(
            create_result_card("Consensus Agreement", consensus_pred, f"{count} out of {total_models} models aligned",
                               consensus_pred), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Comprehensive Model Analysis")

    comparison_df = pd.DataFrame({
        "Model Architecture": list(results.keys()),
        "Diagnostic Prediction": list(results.values()),
        "Confidence Score": [f"{probabilities[n]:.1f}%" if probabilities.get(n) else "N/A" for n in results.keys()],
        "Baseline Accuracy": [MODEL_ACCURACY.get(n, "N/A") for n in results.keys()]
    })


    def highlight_prediction(val):
        if "No ASD" in val:
            return "color: #2e7d32; font-weight: 600;"
        if "ASD Traits Detected" in val:
            return "color: #c62828; font-weight: 600;"
        return ""


    try:
        styled_df = comparison_df.style.map(highlight_prediction, subset=["Diagnostic Prediction"])
    except AttributeError:
        styled_df = comparison_df.style.applymap(highlight_prediction, subset=["Diagnostic Prediction"])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)