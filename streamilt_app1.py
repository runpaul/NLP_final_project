# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:57:24 2024

@author: paulr
"""

import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import pipeline

# Charger le modèle BERT pour l'analyse de sentiment
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, framework="tf")

# Définir l'interface utilisateur Streamlit
st.title("Analyse de Sentiment avec BERT (TensorFlow)")

# Ajouter un champ de texte pour saisir le texte à analyser
text_input = st.text_area("Saisissez votre texte ici:")

# Vérifier si le texte a été saisi
if text_input:
    # Utiliser le modèle pour prédire le sentiment
    result = sentiment_analyzer(text_input)[0]

    # Afficher le résultat
    st.write("Sentiment prédit:", result['label'])
    st.write("Confiance:", result['score'])

# Ajouter des informations sur le modèle utilisé
st.subheader("Modèle utilisé: nlptown/bert-base-multilingual-uncased-sentiment")
st.write("Ce modèle BERT a été pré-entraîné pour l'analyse de sentiment sur des textes multilingues.")

# Ajouter des informations sur l'application
st.subheader("À propos de cette application")
st.write("Cette application utilise le modèle BERT avec TensorFlow pour prédire le sentiment d'un texte saisi par l'utilisateur.")

# Ajouter un lien vers le modèle sur Hugging Face
st.subheader("Plus d'informations sur le modèle")
st.write("[nlptown/bert-base-multilingual-uncased-sentiment sur Hugging Face](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)")

# Ajouter un lien vers le modèle BERT sur Hugging Face
st.subheader("Plus d'informations sur BERT")
st.write("[BERT sur Hugging Face](https://huggingface.co/transformers/model_doc/bert.html)")
