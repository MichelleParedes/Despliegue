import streamlit as st
import pandas as pd
import joblib
import os

# Configuración de la ruta de los archivos
model_path = os.path.join('C:', 'Users', 'Mishu', 'Downloads', 'Despliegue', 'mejor_modelo_log_reg.pkl')
scaler_path = os.path.join('C:', 'Users', 'Mishu', 'Downloads', 'Despliegue', 'scaler.pkl')

# Cargar el modelo y el escalador
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"Error al cargar los archivos: {e}")

def preprocess_data(data):
    # Verificar si el escalador está cargado
    if 'scaler' not in globals():
        st.error("El escalador no está disponible. Asegúrate de que el archivo scaler.pkl esté en la ubicación correcta.")
        return None
    
    try:
        preprocessed_data = scaler.transform(data)
        return preprocessed_data
    except Exception as e:
        st.error(f"Error durante el preprocesamiento: {e}")
        return None

def model_prediction(model, input_data):
    # Verificar si el modelo está cargado
    if 'model' not in globals():
        st.error("El modelo no está disponible. Asegúrate de que el archivo mejor_modelo_log_reg.pkl esté en la ubicación correcta.")
        return None
    
    preprocessed_data = preprocess_data(input_data)
    if preprocessed_data is not None:
        try:
            prediction = model.predict(preprocessed_data)
            return prediction
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
            return None
    else:
        return None

def main():
    st.title("Predicción de Ingreso")
    
    # Ingreso de datos del usuario
    age = st.number_input("Edad", min_value=0, max_value=120, value=30)
    workclass = st.selectbox("Clase de Trabajo", ["Local-gov", "State-gov", "Private", "Self-emp", "Federal-gov"])
    fnlwgt = st.number_input("Peso Final", min_value=10000, max_value=1000000, value=50000)
    
    # Aquí se cerraron las comillas correctamente en la línea del selectbox de Educación
    education = st.selectbox("Educación", ["Bachelors", "Masters", "Doctorate", "HS-grad", "Some-college", "Assoc-acdm", "Assoc-voc", "Prof-school"])
    
    education_num = st.number_input("Número de Educación", min_value=1, max_value=16, value=7)
    marital_status = st.selectbox("Estado Civil", ["Never-married", "Married-civ-spouse", "Divorced", "Widowed", "Separated", "Married-AF-spouse", "Married-spouse-absent"])
    occupation = st.selectbox("Ocupación", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Armed-Forces"])
    relationship = st.selectbox("Relación", ["Other-relative", "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried"])
    race = st.selectbox("Raza", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    sex = st.selectbox("Sexo", ["Male", "Female"])
    capital_gain = st.number_input("Ganancia de Capital", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Pérdida de Capital", min_value=0, max_value=50000, value=0)
    hours_per_week = st.number_input("Horas por Semana", min_value=1, max_value=99, value=40)
    
    # Asegúrate de que el país de origen tenga los nombres correctos
    native_country = st.selectbox("País de Origen", ["United-States", "Mexico", "Germany", "Philippines", "Puerto-Rico", "Canada", "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Japan", "Greece", "Vietnam", "Columbia", "Italy", "Haiti", "Iran", "Nigeria", "Mexico", "Philippines", "Poland", "Portugal", "Brazil", "Hungary", "Taiwan", "Nicaragua", "Scotland", "Ireland", "Cambodia", "Thailand", "Sweden", "France", "Trinidad&Tobago", "Ecuador", "Honduras", "Yugoslavia", "Peru", "Outlying-US(Guam-USVI-etc)", "Hong", "Holand-Netherlands"])
    
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'education-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })
    
    prediction = model_prediction(model, input_data)
    if prediction is not None:
        st.write("Predicción:", prediction[0])

if __name__ == "__main__":
    main()
