import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd

# Load the trained models from the pickle files
with open('st_model.pkl', 'rb') as file:
    stroke_model = pickle.load(file)

with open('nb_model.pkl', 'rb') as file:
    heart_model = pickle.load(file)

with open('kd_model.pkl', 'rb') as file:
    kidney_model = pickle.load(file)

with open('cat_model.pkl', 'rb') as file:
    cancer_model = pickle.load(file)

with open('dy_model.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)


# Define the Streamlit app
def main():
    # Set the app title
    st.title("Disease Prediction")

    # sidebar for navigation
    with st.sidebar:

        selected = option_menu('Multiple Disease Prediction System',

                               ['Diabetes Prediction',
                                'Heart Disease Prediction',
                                'Cancer Risk Prediction',
                                'Kidney Disease Prediction',
                                'Stroke Disease Prediction'],
                               icons=['activity', 'heart', 'cancer', 'kidneys', 'brain'],
                               default_index=0)

    # Diabetes Prediction Page
    if selected == 'Stroke Disease Prediction':

        # page title
        st.title('Stroke Prediction ')
        gender_input = st.selectbox("Gender", ["Male", "Female", "Other"])
        age_input = st.number_input("Age")
        hypertension_input = st.selectbox("Hypertension", ['No', 'Yes'])
        heart_disease_input = st.selectbox("Heart Disease", ['No', 'Yes'])
        ever_married_input = st.selectbox("Ever Married", ["Yes", "No"])
        work_type_input = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked"])
        residence_type_input = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level_input = st.number_input("Average Glucose Level")
        bmi_input = st.number_input("BMI")
        smoking_status_input = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

        if st.button("Predict Stroke"):
            user_data = {
                "Gender": gender_input,
                "Age": age_input,
                "Hypertension": hypertension_input,
                "Heart Disease": heart_disease_input,
                "Ever Married": ever_married_input,
                "Work Type": work_type_input,
                "Residence Type": residence_type_input,
                "Average Glucose Level": avg_glucose_level_input,
                "BMI": bmi_input,
                "Smoking Status": smoking_status_input
            }
            input_data = pd.DataFrame([user_data])
            input_data_encoded = pd.get_dummies(input_data)
            stroke_prediction = stroke_model.predict(input_data_encoded)
            if stroke_prediction[0] == 1:
                st.write("Yes, you are at risk of stroke.")
            else:
                st.write("No, you are not at risk of stroke.")

    if selected == 'Cancer Risk Prediction':

        st.title("Cancer Disease Risk Prediction")
        concave_points_worst = st.number_input("concave points_worst")
        radius_worst = st.number_input("radius_worst")
        perimeter_worst = st.number_input("perimeter_worst")
        area_worst = st.number_input("area_worst")
        concave_points_mean = st.number_input("concave points_mean")
        concavity_mean = st.number_input("concavity_mean")
        concavity_worst = st.number_input("concavity_worst")
        perimeter_mean = st.number_input("perimeter_mean")
        area_mean = st.number_input("area_mean")
        area_se = st.number_input("area_se")

        if st.button("Predict Cancer Risk"):
            user_data = {
                "concave points_worst": concave_points_worst,
                "radius_worst": radius_worst,
                "perimeter_worst": perimeter_worst,
                "area_worst": area_worst,
                "concave points_mean": concave_points_mean,
                "concavity_mean": concavity_mean,
                "concavity_worst": concavity_worst,
                "perimeter_mean": perimeter_mean,
                "area_mean": area_mean,
                "area_se": area_se
            }
            input_data = pd.DataFrame([user_data])
            cancer_prediction = cancer_model.predict(input_data)

            if cancer_prediction > 0.80:
                st.write("High risk - Consult With Doctor Immediately .")
            elif cancer_prediction >= 0.40 and cancer_prediction <= 0.80:
                st.write("Normal -  Take precautions.")
            else:
                st.write("Safe.")

    if selected == 'Heart Disease Prediction':

        st.title("Heart Disease Prediction")
        age = st.number_input("Age")
        gender = st.number_input("Gender(0-Female,1-Male")
        cp = st.number_input("Chest Pain Type")
        trestbps = st.number_input("Resting Blood Pressure")
        chol = st.number_input("Cholesterol")
        fbs = st.number_input("Fasting Blood Sugar")
        restecg = st.number_input("Resting ECG")
        thalach = st.number_input("Maximum Heart Rate Achieved")
        exang = st.number_input("Exercise-Induced Angina")
        oldpeak = st.number_input("ST Depression Induced by Exercise")
        slope = st.number_input("Slope of the Peak Exercise ST Segment")
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy")
        thal = st.number_input("Thalassemia")

        if st.button("Predict Heart Disease"):
            user_data = {
                "age": int(age),
                "gender": int(gender),
                "cp": float(cp),
                "trestbps": float(trestbps),
                "chol": float(chol),
                "fbs": float(fbs),
                "restecg": float(restecg),
                "thalach": float(thalach),
                "exang": float(exang),
                "oldpeak": float(oldpeak),
                "slope": float(slope),
                "ca": float(ca),
                "thal": float(thal)
            }

            input_data = pd.DataFrame([user_data])
            heart_prediction = heart_model.predict(input_data)

            if heart_prediction[0] == 1:
                st.write("Yes, you are at risk of Heart Disease.")
            else:
                st.write("No, you are not at risk of Heart Disease.")

    if selected == 'Kidney Disease Prediction':

        st.title("Kidney Disease Prediction")
        age = st.number_input("Age")
        blood_pressure = st.number_input("Blood Pressure")
        specific_gravity = st.number_input("Specific Gravity")
        albumin = st.number_input("Albumin")
        sugar = st.number_input("Sugar")
        red_blood_cells = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pus_cell = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pus_cell_clumps = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
        bacteria = st.selectbox("Bacteria", ["notpresent", "present"])
        blood_glucose_random = st.number_input("Blood Glucose Random")
        blood_urea = st.number_input("Blood Urea")
        serum_creatinine = st.number_input("Serum Creatinine")
        sodium = st.number_input("Sodium")
        potassium = st.number_input("Potassium")
        hemoglobin = st.number_input("Hemoglobin")
        hypertension = st.selectbox("Hypertension", ["yes", "no"])
        diabetes_mellitus = st.selectbox("Diabetes Mellitus", ["yes", "no"])
        coronary_artery_disease = st.selectbox("Coronary Artery Disease", ["yes", "no"])
        appetite = st.selectbox("Appetite", ["good", "poor"])
        pedal_edema = st.selectbox("Pedal Edema", ["yes", "no"])
        anemia = st.selectbox("Anemia", ["yes", "no"])

        if st.button("Predict Kidney Disease"):
            user_data = {
                "age": age,
                "blood_pressure": blood_pressure,
                "specific_gravity": specific_gravity,
                "albumin": albumin,
                "sugar": sugar,
                "red_blood_cells": red_blood_cells,
                "pus_cell": pus_cell,
                "pus_cell_clumps": pus_cell_clumps,
                "bacteria": bacteria,
                "blood_glucose_random": blood_glucose_random,
                "blood_urea": blood_urea,
                "serum_creatinine": serum_creatinine,
                "sodium": sodium,
                "potassium": potassium,
                "hemoglobin": hemoglobin,
                "hypertension": hypertension,
                "diabetes_mellitus": diabetes_mellitus,
                "coronary_artery_disease": coronary_artery_disease,
                "appetite": appetite,
                "pedal_edema": pedal_edema,
                "anemia": anemia
            }
            input_data = pd.DataFrame([user_data])
            input_data_encoded = pd.get_dummies(input_data)
            kidney_prediction = kidney_model.predict(input_data_encoded)
            if kidney_prediction[0] == 1:
                st.write("Yes, you are at risk of Kidney Disease.")
            else:
                st.write("No, you are not at risk of Kidney Disease.")

    if selected == 'Diabetes Prediction':

        st.title("Diabetes Prediction")
        pregnancies = st.number_input("Pregnancies")
        glucose = st.number_input("Glucose")
        blood_pressure = st.number_input("Blood Pressure")
        skin_thickness = st.number_input("Skin Thickness")
        insulin = st.number_input("Insulin")
        bmi = st.number_input("BMI")
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function")
        age = st.number_input("Age")

        if st.button("Predict Diabetes"):
            user_data = {
                "Pregnancies": int(pregnancies),
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": diabetes_pedigree_function,
                "Age": int(age)
            }
            input_data = pd.DataFrame([user_data])
            diabetes_prediction = diabetes_model.predict(input_data)
            if diabetes_prediction[0] == 1:
                st.write("Yes, you Have Diabetes.")
            else:
                st.write("No, you Don't Have Diabetes.")


# Run the app
if __name__ == '__main__':
    main()
