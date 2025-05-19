# Star-Galaxy Classifier

A Streamlit-based machine learning app for classifying astronomical objects (stars vs galaxies) using SDSS (Sloan Digital Sky Survey) data. Built to demonstrate a full ML workflow: data preprocessing, model training, evaluation, and interactive prediction.

---

## Features

- Upload SDSS feature data (CSV)
- Train/test multiple models (Logistic Regression, Random Forest, SVM, etc.)
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix
- Perform k-fold cross-validation
- Manually input feature values for real-time predictions
- Visualize decision boundaries (2D projection)
- Streamlit UI for interactive exploration

---

## Tech Stack

- Python, Streamlit  
- Scikit-learn (models + preprocessing)  
- Pandas, NumPy, Matplotlib/Seaborn  

---

##  Getting Started

```bash
git clone https://github.com/yourusername/star-galaxy-classifier
cd star-galaxy-classifier
pip install -r requirements.txt
streamlit run app.py
