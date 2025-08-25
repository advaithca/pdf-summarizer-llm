# 📄 PDF Summarizer with LLaMA + Streamlit

This project is a simple **offline PDF summarization app** built using [Streamlit](https://streamlit.io/) and [Ollama](https://ollama.com/).  
It allows you to upload one or more PDF files, process their contents, and generate summaries using the **LLaMA 3.2 (3.2B)** model with GPU acceleration.

---

## 🚀 Features
- Upload **single or multiple PDFs**.
- Extracts and cleans text from PDFs.
- Summarizes content using **Ollama LLaMA models**.
- Runs **completely offline** (air-gapped friendly).
- GPU support (if available).

---

## 🛠 Installation

### 1. Clone this repository  
```bash```
```git clone https://github.com/your-username/pdf-summarizer-llama.git```
```cd pdf-summarizer-llama ```

### 2. Install dependencies

Make sure you have Python 3.9+ installed (Tested on 3.12).

```pip install -r requirements.txt```


Your requirements.txt should contain:

```streamlit```
```pypdf```
```ollama```

### 3. Install Ollama

Follow the official instructions for your OS.

### 4. Download the LLaMA model
```ollama pull llama3.2```


Confirm it’s available:

```ollama list```

▶️ Usage

Run the Streamlit app:

```streamlit run app.py```


Then open your browser at:

http://localhost:8501


Upload your PDFs and get instant summaries!

📂 Project Structure
pdf-summarizer-llama/
│── app.py              # Main Streamlit app
│── requirements.txt    # Python dependencies
│── README.md           # Documentation

⚡ Notes

Works offline, no internet required after setup.

GPU is automatically used if available.

For large PDFs, summarization may take some time.