# Gita Guide: RAG chatbot (ChromaDB + Groq + Streamlit)

## How it works
PDF -> text -> chunks -> embeddings (local model) -> ChromaDB.
Question -> embed -> find the 5 closest chunks -> send them + question to Groq LLM -> streamed answer.

## Setup (VS Code terminal)
1. Unzip, open the folder in VS Code (File > Open Folder).
2. Create a virtual environment
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. `pip install -r requirements.txt`  (takes a few minutes)
4. Copy `.env.example` to `.env` and paste your Groq key after `GROQ_API_KEY=`
5. Put your PDF in the `data` folder.
6. `python ingest.py`   (builds the vector DB, once)
7. `streamlit run app.py`  (opens the chat in your browser)

## Notes
- Use Python 3.10 to 3.12 for best compatibility with ChromaDB.
- If ingest says "almost no text found", the PDF is scanned images and needs OCR first.
- Change the PDF? Re-run `python ingest.py`.
- If Groq retires the model, set GROQ_MODEL in `.env` to a current one from console.groq.com.
