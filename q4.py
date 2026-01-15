import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize

# Download the punkt model
nltk.download('punkt')

st.title("Text Chunking Web App")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "

    st.subheader("Sample text (indices 58-68)")
    st.write(text[58:68])

    # Sentence tokenization
    sentences = sent_tokenize(text)
    st.subheader("Tokenized Sentences")
    st.write(f"Total sentences: {len(sentences)}")
    for i, s in enumerate(sentences[:10], 1):
        st.write(f"{i}. {s}")
