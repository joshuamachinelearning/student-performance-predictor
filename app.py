import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Load model and artifacts ---
model = joblib.load('best_model.pkl')
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
scaler = joblib.load('scaler.pkl')

# --- Page setup ---
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")
st.markdown("Predict student academic performance and review model test accuracy.")

# --- Feature Descriptions ---
with st.expander("ℹ️ Feature Descriptions and Codes", expanded=True):
    st.markdown("""
    - **Gender**  
      - `0`: Male  
      - `1`: Female  

    - **ParentalEducation** (Parent's highest education level)  
      - `0`: None  
      - `1`: High School  
      - `2`: Some College  
      - `3`: Bachelor's  
      - `4`: Higher  

    - **StudyTimeWeekly**  
      - Weekly study time in hours (range: 0 to 20)  

    - **Absences**  
      - Number of absences during the school year (range: 0 to 30)  

    - **Tutoring**  
      - `0`: No  
      - `1`: Yes  

    - **ParentalSupport** (Level of parental support)  
      - `0`: None  
      - `1`: Low  
      - `2`: Moderate  
      - `3`: High  
      - `4`: Very High  

    - **Extracurricular**  
      - `0`: No  
      - `1`: Yes  

    - **Sports**  
      - `0`: No  
      - `1`: Yes  

    - **Music**  
      - `0`: No  
      - `1`: Yes  

    - **Volunteering**  
      - `0`: No  
      - `1`: Yes  
    """)

# --- Features and label map ---
feature_names = ['Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
                 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

# --- Prediction function ---
def predict_grade(df):
    df_scaled = df.copy()
    df_scaled[['StudyTimeWeekly', 'Absences']] = scaler.transform(df_scaled[['StudyTimeWeekly', 'Absences']])
    return model.predict(df_scaled)

# --- User input form ---
st.header("📥 Predict a Single Student's Grade")

with st.form("prediction_form"):
    user_data = {
        'Gender': st.selectbox("Gender", [0, 1]),
        'ParentalEducation': st.selectbox("Parental Education (0=None to 4=Higher)", list(range(5))),
        'StudyTimeWeekly': st.slider("Study Time Weekly (hours)", 0, 20, 5),
        'Absences': st.slider("Number of Absences", 0, 30, 2),
        'Tutoring': st.selectbox("Tutoring?", [0, 1]),
        'ParentalSupport': st.selectbox("Parental Support (0=None to 4=Very High)", list(range(5))),
        'Extracurricular': st.selectbox("Extracurricular?", [0, 1]),
        'Sports': st.selectbox("Sports?", [0, 1]),
        'Music': st.selectbox("Music?", [0, 1]),
        'Volunteering': st.selectbox("Volunteering?", [0, 1])
    }

    submit = st.form_submit_button("Predict")

# --- Predict and display result ---
if submit:
    input_df = pd.DataFrame([user_data])
    pred_class = predict_grade(input_df)[0]
    input_df['Predicted Grade'] = label_map[pred_class]

    st.success(f"✅ Predicted Grade: **{label_map[pred_class]}**")
    st.write("### 🔎 Input Summary with Prediction")
    st.dataframe(input_df)

# --- Model performance on test set ---
st.divider()
st.header("📊 Model Performance on Test Set")

# Predict test set
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
st.metric(label="✅ Accuracy", value=f"{acc:.2%}")

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.subheader("📄 Classification Report")
st.dataframe(report_df.style.format(precision=2))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

st.caption("Model evaluated on holdout test set.")