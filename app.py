import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
from scripts.run_inference import run_inference
from src.data.preprocessing import process_sdss_data
from src.data.sdss_query import get_sdss_data
import matplotlib.pyplot as plt

def highlight_correct_wrong(row):
    if row['Actual Class'] == row['Prediction']:
        return ['background-color: #a3f7b5; color: black'] * len(row)  # green
    else:
        return ['background-color: #f7a3a3; color: black'] * len(row)  # red

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

st.title("🌌 Star-Galaxy Classifier 🌌")

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
    # Display the first 20 rows of the comparison
    st.dataframe(
        comparison.head(200).style.apply(highlight_correct_wrong, axis=1),
        use_container_width=True
    )
    # Allow the user to download the predictions
    st.download_button('Download full observation data', results.to_csv(index=False), 'predictions.csv')

    # Count correct vs incorrect predictions
    correct = (comparison['Actual Class'] == comparison['Prediction']).sum()
    incorrect = len(comparison) - correct

    # Create pie chart
    fig, ax = plt.subplots()
    ax.pie(
        [correct, incorrect],
        labels=[f'{correct} Correct', f'{incorrect} Incorrect'],
        colors=['#a3f7b5', '#f7a3a3'],  # green and red
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')  # Equal aspect ratio makes the pie round

    st.pyplot(fig)

    # Calculate metrics
    total = len(comparison)
    correct = (comparison['Actual Class'] == comparison['Prediction']).sum()
    incorrect = total - correct
    accuracy = correct / total * 100

    # Display metrics side by side
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total)
    col2.metric("Correct", correct)
    col3.metric("Accuracy", f"{accuracy:.2f}%")

    # Show explanation for a random sample prediction
    random_sample = results.sample(1)  # Random row from results
    sample_features = random_sample.drop(columns=['Actual Class', 'Prediction'])
    explanation = model.predict_proba(sample_features)[0]  # Get probability of class predictions

    st.subheader("Model Insights: Prediction Breakdown for a Random Sample")
    st.write("Random Sample Features:")
    st.dataframe(sample_features)
    st.write(f"Prediction Probabilities:\nStar: {explanation[1] * 100:.2f}% | Galaxy: {explanation[0] * 100:.2f}%")
    st.write(f"Predicted Class: {random_sample['Prediction'].values[0]}")

# Manual input
st.subheader('Or input your own sky survey data! :)')
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

