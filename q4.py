import streamlit as st
from PyPDF2 import PdfReader
import nltk
import os
from nltk.tokenize import sent_tokenize

# ---------------------- NLTK Setup ----------------------
# Create local nltk data folder (needed for Streamlit Cloud)
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# ---------------------- App Config ----------------------
st.set_page_config(page_title="Text Chunker", layout="wide")
st.title("Text Chunker (Word-based & Sentence-based)")

# ---------------------- Text Input ----------------------
st.write(
    "This app can split text into chunks based on **number of words** "
    "or **semantic sentences** using NLTK."
)

# Option to upload PDF
uploaded_file = st.file_uploader("Upload PDF (optional)", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
else:
    text = st.text_area(
        "Input text",
        value=(
            "Ku Muhammad Na'im Ku Khalif is actively involved in teaching, research, and "
            "community outreach. He regularly works with AI, data science, and Internet "
            "safety campaigns across schools in Pahang, collaborating with various agencies."
        ),
        height=200,
    )

# ---------------------- Word Chunking ----------------------
def chunker(input_data: str, N: int):
    input_words = input_data.split()
    output = []
    cur_chunk = []
    count = 0
    for word in input_words:
        cur_chunk.append(word)
        count += 1
        if count == N:
            output.append(" ".join(cur_chunk))
            count, cur_chunk = 0, []
    if cur_chunk:
        output.append(" ".join(cur_chunk))
    return output

chunk_size = st.number_input(
    "Number of words per chunk (N)",
    min_value=1,
    max_value=2000,
    value=20,
    step=1,
)

# ---------------------- Sentence Tokenization ----------------------
st.write("---")
st.subheader("Sentence-based Chunking")
do_sent_chunking = st.checkbox("Enable sentence-based chunking using NLTK", value=True)

# ---------------------- Process Button ----------------------
if st.button("Create chunks"):
    if not text.strip():
        st.warning("Please provide some text to chunk.")
    else:
        # Word-based chunking
        chunks = chunker(text, int(chunk_size))
        st.success(f"Number of word-based chunks = {len(chunks)}")

        idx = st.number_input(
            "Select word-chunk index to view",
            min_value=1,
            max_value=len(chunks),
            value=1,
            step=1,
        )
        st.subheader(f"Word-based Chunk {idx}")
        st.write(chunks[idx - 1])

        with st.expander("Show all word-based chunks"):
            for i, ch in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(ch)
                st.markdown("---")

        # Sentence-based chunking
        if do_sent_chunking:
            sentences = sent_tokenize(text)
            st.success(f"Number of sentences = {len(sentences)}")
            idx_sent = st.number_input(
                "Select sentence index to view",
                min_value=1,
                max_value=len(sentences),
                value=1,
                step=1,
            )
            st.subheader(f"Sentence {idx_sent}")
            st.write(sentences[idx_sent - 1])

            with st.expander("Show all sentences"):
                for i, s in enumerate(sentences, start=1):
                    st.markdown(f"**Sentence {i}**")
                    st.write(s)
                    st.markdown("---")
