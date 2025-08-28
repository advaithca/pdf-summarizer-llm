import streamlit as st
import fitz  # PyMuPDF for PDFs
import pandas as pd
import ollama

# ------------------------
# File Loaders
# ------------------------
def load_file(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    elif file.type in ["text/plain", "application/txt"]:
        return file.read().decode("utf-8")

    elif file.type in ["text/csv", "application/vnd.ms-excel"]:
        df = pd.read_csv(file)
        return df.to_string()

    else:
        return "⚠️ Unsupported file type"


# ------------------------
# Streamlit App
# ------------------------
st.title("Chat with Multiple Files (Ollama-powered)")

# Get available models from Ollama
models_resp = ollama.list()
models = [m.model for m in models_resp.models]

if not models:
    st.error("⚠️ No Ollama models found. Run `ollama pull llama3` (or another model) to use this app.")
    st.stop()

model_choice = st.selectbox("Choose Ollama model:", models)

# File uploader
uploaded_files = st.file_uploader(
    "Upload files (PDF, TXT, CSV)", type=["pdf", "txt", "csv"], accept_multiple_files=True
)

# Session memory (all files in one session, but kept separated)
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_files:
    # Load and structure content per file
    file_contexts = {}
    for file in uploaded_files:
        if file.name not in file_contexts:
            file_contexts[file.name] = load_file(file)

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask about your files..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            # System prompt ensures separation between files
            system_prompt = (
                "You are an assistant that answers questions about multiple files. "
                "Each file must be treated separately. "
                "If a question is about one file, answer only from that file. "
                "If it involves multiple files, answer per-file without mixing their contents.\n\n"
            )

            file_prompts = "\n\n".join(
                [f"--- Start of {fname} ---\n{content}\n--- End of {fname} ---"
                for fname, content in file_contexts.items()]
            )

            print(file_prompts)
            stream = ollama.chat(
                model=model_choice,
                messages=[
                    {"role": "system", "content": system_prompt + file_prompts},
                    *st.session_state.messages
                ],
                stream=True,
            )

             # ✅ This part fixes your issue
            response_text = ""
            placeholder = st.empty()
            for chunk in stream:
                if hasattr(chunk, "message") and chunk.message and chunk.message.content:
                    response_text += chunk.message.content
                    placeholder.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})