import streamlit as st
import fitz  # PyMuPDF
import ollama

# ----------------------------
# Utility: Extract text from PDFs
# ----------------------------
def extract_text_from_pdf(file):
    """Extract text from a single PDF file."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


# ----------------------------
# Summarize text with Ollama
# ----------------------------
def summarize_text(text, model="llama3.2:latest", max_words=500):
    """Send extracted text to Ollama for summarization."""
    prompt = f"Summarize the following text in about {max_words} words:\n\n{text}"
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF Summarizer (Offline)", layout="wide")
st.title("📄 PDF Summarizer (Offline with Ollama + LLaMA)")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

max_words = st.slider("Summary length (approx. words)", 100, 1000, 400, 50)

# Get available models from Ollama
try:
    available_models = [m.model for m in ollama.list().models]
except Exception as e:
    available_models = []
    st.error(f"Could not fetch models from Ollama: {e}")

if available_models:
    model_name = st.selectbox("Choose Ollama model", available_models, index=0)
else:
    model_name = None
    st.warning("⚠️ No models found. Make sure you have pulled or created models with `ollama pull` or `ollama create`.")

if uploaded_files and model_name and st.button("Summarize"):
    all_texts = []
    for file in uploaded_files:
        st.write(f"📖 Extracting text from **{file.name}** ...")
        text = extract_text_from_pdf(file)
        all_texts.append(text)

    combined_text = "\n\n".join(all_texts)

    with st.spinner(f"Summarizing with {model_name}..."):
        summary = summarize_text(combined_text, model=model_name, max_words=max_words)

    st.subheader("📝 Summary")
    st.write(summary)
