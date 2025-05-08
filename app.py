import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from scripts.run_inference import run_inference
from src.data.preprocessing import process_sdss_data
from src.data.sdss_query import get_sdss_data
import matplotlib.pyplot as plt

# Load available models
MODEL_DIR = Path("models")
model_files = {
    "Logistic Regression": MODEL_DIR / "LR_best_model.pkl",
    "Random Forest": MODEL_DIR / "RF_best_model.pkl",
    "XGBoost": MODEL_DIR / "XGB_best_model.pkl",
    "SVM": MODEL_DIR / "SVM_best_model.pkl"
}

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

st.markdown("""
# 🌌 Star/Galaxy Classifier 🌌

Welcome to the **Star-Galaxy Classifier**! This app uses machine learning models to classify objects in the sky as either **stars** or **galaxies** based on real SDSS survey measurements!

### How to Use:
1. **Choose a Model**: Select one of the pre-trained models (Logistic Regression, Random Forest, XGBoost, or SVM) in the sidebar.
2. **Fetch Data**: Use the slider to set the number of SDSS samples to retrieve. Click **'Run Inference'** to retrieve and classify the data.
3. **View Results**: After running the inference, you will see:
    - **Model Evaluation**: Accuracy, F1 Score, and Cross-Validation details.
    - **Prediction Breakdown**: A random sample will be shown with its features and prediction probability.
    - **Download Results**: Download the results (with all features, prediction, and true class) as a CSV file.
4. **Manual Input**: You can also manually input your own sky survey data to classify individual samples as stars or galaxies.

For more details, explore the insights and predictions provided for the data!

""")

# Sidebar
model_name = st.sidebar.selectbox('Choose a model', list(model_files.keys()))
sample_size = st.sidebar.slider('Number of untested SDSS Samples to retrieve', 100, 50000, 1000, step=100)

# Load model
model = load_model(model_files[model_name])

# Fetch data
st.subheader('Querying SDSS...')
try:
    raw_data = get_sdss_data(offset=20000, limit=sample_size)
    data = process_sdss_data(raw_data)
    st.success(f'Fetched {len(data)} samples from SDSS.')
except Exception as e:
    st.error(f'Failed to fetch data: {e}')
    st.stop()

# Run inference
if st.button('Run Inference'):
    preds = run_inference(model, data)
    results = data.copy()
    results['Prediction'] = preds

    # Decode 'prediction' from binary to 'star'/'galaxy'
    results['Prediction'] = results['Prediction'].replace({0: 'Galaxy', 1: 'Star'})
    # Rename 'class' column to 'Actual Class'
    results = results.rename(columns={'class': 'Actual Class'})
    # Create a simplified view for comparison: prediction vs actual
    comparison = results[['Actual Class', 'Prediction']]
    # Allow the user to download the predictions
    st.download_button('Download full observation data', results.to_csv(index=False), 'predictions.csv')

    # Count correct vs incorrect predictions
    correct = (comparison['Actual Class'] == comparison['Prediction']).sum()
    incorrect = len(comparison) - correct
    fig, ax = plt.subplots()
    ax.pie(
        [correct, incorrect],
        labels=[f'{correct} Correct', f'{incorrect} Incorrect'],
        colors=['#a3f7b5', '#f7a3a3'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')
    st.pyplot(fig)

    # Calculate metrics
    total = len(comparison)
    correct = (comparison['Actual Class'] == comparison['Prediction']).sum()
    incorrect = total - correct
    accuracy = correct / total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total)
    col2.metric("Correct", correct)
    col3.metric("Accuracy", f"{accuracy:.1f}%")

    st.subheader('Model Insights: Evaluation Metrics')
    # F1 Score
    try:
        f1 = f1_score(comparison['Actual Class'], comparison['Prediction'], pos_label='Star')
        st.write(f"F1 Score (Star = Positive): {f1:.2f}")
    except Exception as e:
        st.warning(f"Could not compute F1 Score: {e}")
    # Accuracy
    accuracy = accuracy_score(comparison['Actual Class'], comparison['Prediction'])
    st.write(f"Accuracy: {accuracy:.2%}")
    # Cross-Val
    try:
        label_data = data['class']
        if label_data.dtype == 'object':
            label_data = LabelEncoder().fit_transform(label_data)
        cv_score = cross_val_score(model, data.drop(columns=['class']), label_data, cv=5)
        st.write(f"Cross-Validation Accuracy: {cv_score.mean():.2f} ± {cv_score.std():.4f}")
    except Exception as e:
        st.warning(f"Cross-validation failed: {e}")

    st.subheader("Model Insights: Prediction Breakdown for a Random Sample")
    # Show explanation for a random sample prediction
    random_sample = results.sample(1)  # Random row from results
    sample_features = random_sample.drop(columns=['Actual Class', 'Prediction'])
    st.write("Random Sample Features:")
    st.dataframe(sample_features)
    try:
        if hasattr(model, "predict_proba"):
            explanation = model.predict_proba(sample_features)[0]
            st.write(
                f"Prediction Probabilities:\nStar: {explanation[1] * 100:.2f}% | Galaxy: {explanation[0] * 100:.2f}%")
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(sample_features)[0]
            st.write(f"Decision Score: {decision:.4f} (positive means Star)")
        else:
            st.write("This model does not support probability or decision explanations.")
    except Exception as e:
        st.warning(f"Could not extract prediction explanation: {e}")

    st.write(f"Predicted Class: {random_sample['Prediction'].values[0]}")

# Manual input
st.subheader("Or input your own sky survey data and see if you're looking at a star or a galaxy! :)")
with st.form('manual_input_form'):
    input_data = {}
    for col in [
        'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z',
        'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z',
        'petroR50_r', 'petroR90_r'
    ]:
        input_data[col] = st.number_input(col, value=0.0)
    submitted = st.form_submit_button('Predict manually')

if submitted:
    df = pd.DataFrame([input_data])
    df = process_sdss_data(df)
    pred = model.predict(df)[0]
    label = "Galaxy" if pred == 0 else "Star" if pred == 1 else str(pred)
    st.success(f'Predicted Class: {label}')

