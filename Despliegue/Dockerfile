# Usar la imagen base de Python
FROM python:3.9-slim

# Crear y usar el directorio de trabajo
WORKDIR /app

# Copiar los archivos de la aplicación al contenedor
COPY app.py requirements.txt mejor_modelo_log_reg.pkl data_evaluacion.csv ./

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que la aplicación usará
EXPOSE 8501

# Comando para ejecutar la aplicación de Streamlit
CMD ["streamlit", "run", "app.py"]
