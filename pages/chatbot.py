import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializar lemmatizer y cargar el modelo y datos
lemmatizer = WordNetLemmatizer()
with open('./pages/sources/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('./pages/sources/words.pkl', 'rb'))
classes = pickle.load(open('./pages/sources/classes.pkl', 'rb'))
model = load_model('./pages/sources/chatbot_model.h5')

# Funciones de preprocesamiento y predicción


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category


def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Lo siento, no entiendo tu pregunta."


# Configuración de la interfaz de usuario con Streamlit
st.title("Chatbot Interactivo")
st.write("Pregunta lo que quieras y el chatbot te responderá.")

# Mantener el historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Primer mensaje del asistente
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿cómo puedo ayudarte?")
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?"})

# Entrada del usuario
if user_input := st.chat_input("Escribe tu mensaje aquí..."):
    # Mostrar el mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Obtener la respuesta del chatbot
    response_tag = predict_class(user_input)
    response = get_response(response_tag, intents)

    # Mostrar la respuesta del chatbot
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
