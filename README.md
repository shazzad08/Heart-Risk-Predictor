# ❤️ Heart Disease Risk Predictor

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An AI-powered early heart disease risk screening system. This project uses a Machine Learning classification model (K-Nearest Neighbors) to assess a patient's risk of heart disease based on their medical information and vitals. It features a beautiful, responsive, and user-friendly web interface built with Streamlit.
📍 Live: https://heart-risk-predictor-d4vubmp43y8e7bs98eieed.streamlit.app/

## ✨ Features

- **Accurate Predictions**: Uses a trained K-Nearest Neighbors (KNN) model to predict the probability of heart disease.
- **Interactive UI**: A modern and intuitive dashboard built with Streamlit and custom CSS for a premium user experience.
- **Detailed Health Overview**: Displays inputted patient data in clean, easy-to-read cards.
- **Actionable Recommendations**: Provides tailored medical advice and heart-health tips depending on whether the predicted risk is high or low.
- **Risk Probability Indicator**: Visualizes the predicted risk percentage with a color-coded gradient bar.

## 🛠️ Tech Stack

- **Frontend/Backend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: `scikit-learn` (KNN Classifier)
- **Data Manipulation**: `pandas`, `numpy`
- **Model Serialization**: `joblib`

## 📂 Project Structure

```text
Heart-Risk-Predictor/
│
├── app.py                 # Main Streamlit application
├── Project.ipynb          # Jupyter Notebook with EDA and model training
├── heart.csv              # Heart Failure Prediction Dataset
├── KNN_heart.pkl          # Serialized K-Nearest Neighbors model
├── scaler.pkl             # Serialized data scaler for preprocessing
├── columns.pkl            # Serialized list of expected columns for the model
└── README.md              # Project documentation
```

## 📊 Dataset Information

The model is trained on a comprehensive heart disease dataset (`heart.csv`) containing 918 entries with 11 clinical features:
- **Age**: Age of the patient (years)
- **Sex**: Male (M) / Female (F)
- **ChestPainType**: Chest pain type (ATA, NAP, TA, ASY)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol (mg/dL)
- **FastingBS**: Fasting blood sugar > 120 mg/dL
- **RestingECG**: Resting electrocardiogram results (Normal, ST, LVH)
- **MaxHR**: Maximum heart rate achieved (bpm)
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: The slope of the peak exercise ST segment (Up, Flat, Down)

## 🚀 Installation & Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd Heart-Risk-Predictor
   ```

2. **Install the required dependencies**:
   Make sure you have Python 3.8+ installed. You can install the required packages using `pip`:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the Web App**:
   Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## 🩺 Usage

1. Open the web app in your browser.
2. In the sidebar, fill in the **Patient Information** using the sliders and dropdown menus.
3. Click the **Predict Risk** button.
4. Review the prediction results, risk probability, health overview, and the personalized recommendations provided on the main dashboard.

## 👨‍💻 Author

Built with Streamlit & Machine Learning | Made by **Shazzad**
