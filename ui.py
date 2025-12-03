import streamlit as st
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from utils.ui_utils import draw_confidence_score

MODEL_DIR = "models/"  # путь к распакованной папке

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
st.title('Data-assessment PUI')

with st.form('Check Sentiment'):
    col_text, col_button = st.columns([4, 1], vertical_alignment='bottom')

    with col_text:
        user_text = st.text_input("Введите текст:", "")

    with col_button:
        run = st.form_submit_button("Проверить")

    if run and user_text.strip():
        res = clf(user_text)

        label = res[0]['label']
        score = res[0]['score']

        name, chart = st.columns([1, 4], vertical_alignment='center')
        with name:
            st.subheader(label.upper())

        with chart:
            st.plotly_chart(draw_confidence_score(score), use_container_width=True)
        
         