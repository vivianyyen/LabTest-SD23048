# app.py
import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize

# ----------------------------
# Step 0: NLTK download
# ----------------------------
nltk.download('punkt')

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF Text Chunking", layout="wide")
st.title("Text Chunking Web App using NLTK Sentence Tokenizer")
st.caption("Extract text from PDF, split into sentences, and display semantic chunks.")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Step 2: Extract text
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    
    st.subheader("Sample of Extracted Text (indices 58–68)")
    # Step 3: Display sample
    sample_text = text[58:68]
    st.write(sample_text)

    # Step 4: Sentence tokenization
    sentences = sent_tokenize(text)
    st.subheader("Sentence Tokenized Text (Semantic Chunks)")
    st.write(f"Total sentences found: {len(sentences)}")
    
    # Display first 10 chunks for illustration
    st.write("Sample sentences (first 10):")
    for i, sentence in enumerate(sentences[:10], start=1):
        st.write(f"{i}. {sentence}")

else:
    st.info("Upload a PDF file to see text chunking results.")
