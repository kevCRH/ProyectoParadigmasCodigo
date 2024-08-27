import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
import textwrap

# Cargar variables de entorno
load_dotenv()

# Configura tu clave de API de Google Gemini
genai.configure(api_key=os.getenv('API_KEY'))

# Crear el modelo de Gemini
model_name = 'gemini-1.5-flash'  # Cambia según el modelo que prefieras
model = genai.GenerativeModel(model_name)


# Función para convertir texto a formato Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)


# Función para procesar la solicitud del usuario usando Gemini
def procesar_solicitud(texto, solicitud):
    try:
        # Enviar mensaje y obtener respuesta
        response = model.generate_content(f"{solicitud}\n\nTexto:\n{texto}")

        # Convertir la respuesta a Markdown y retornar
        return to_markdown(response.text)
    except Exception as e:
        if "RateLimitError" in str(e):
            st.error(
                "Se ha superado el límite de cuota. Por favor, inténtelo de nuevo más tarde.")
            time.sleep(60)  # Espera de 60 segundos antes de reintentar
        else:
            st.error(f"Ocurrió un error con la API de Gemini: {e}")
        return "No se pudo procesar la solicitud."


# Título de la aplicación
st.title("Extracción y Procesamiento de Texto de Imágenes")
st.sidebar.success("Menu")

# Subir imagen
uploaded_file = st.file_uploader(
    "Elige una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Leer la imagen
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Convertir a escala de grises
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Ajustar contraste y brillo
        contrast = 1.0  # Contraste (0-127)
        brightness = 40  # Brillo (0-100)
        adjusted_image = cv2.addWeighted(
            gray_image, contrast, gray_image, 0, brightness)

        # Convertir la imagen ajustada a formato PIL para pytesseract
        adjusted_image_pil = Image.fromarray(adjusted_image)

        # Configuración personalizada de PSM (Page Segmentation Mode)
        custom_psm_config = r'--psm 4'

        # Extraer texto usando pytesseract
        tesseract_response = pytesseract.image_to_string(
            adjusted_image_pil, config=custom_psm_config)

        # Mostrar imágenes
        st.image(image, caption='Imagen Original', use_column_width=True)
        st.image(adjusted_image, caption='Imagen Ajustada',
                 use_column_width=True)

        # Mostrar el texto extraído
        st.subheader("Texto Extraído")
        st.text(tesseract_response)

        # Solicitar al usuario una instrucción para procesar el texto
        solicitud = st.text_area("¿Qué deseas que haga con el texto extraído?")

        if st.button("Enviar Solicitud"):
            if solicitud:
                respuesta = procesar_solicitud(tesseract_response, solicitud)
                st.subheader("Respuesta de la IA")
                st.markdown(respuesta)  # Mostrar respuesta en formato Markdown
            else:
                st.error("Por favor, ingresa una instrucción para la IA.")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la imagen: {e}")
