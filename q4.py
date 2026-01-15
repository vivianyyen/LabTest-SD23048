import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import os

# ----------------------
# Step 0: Setup NLTK for cloud
# ----------------------
# Create a local folder for nltk data
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Download punkt into this folder (works even in Streamlit Cloud)
nltk.download("punkt", download_dir=nltk_data_dir)

# Tell NLTK to look here first
nltk.data.path.append(nltk_data_dir)

# ----------------------
# Streamlit UI
# ----------------------
st.title("PDF Text Chunking Web App")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    st.subheader("Sample text (indices 58–68)")
    st.write(text[58:68])

    # ----------------------
    # Sentence tokenization
    # ----------------------
    sentences = sent_tokenize(text)
    st.subheader("Sentence-based Chunks")
    st.write(f"Total sentences found: {len(sentences)}")

    # Display first 10 sentences
    for i, s in enumerate(sentences[:10], start=1):
        st.write(f"{i}. {s}")

else:
    st.info("Upload a PDF to see sentence chunks.")
