"""Step 2: chat interface.   Run:  streamlit run app.py"""
import streamlit as st

import config
from rag import GitaRAG

st.set_page_config(page_title="Gita Guide", page_icon="🪷", layout="centered")


@st.cache_resource(show_spinner="Loading model and vector database...")
def get_rag() -> GitaRAG:
    return GitaRAG()


st.title("🪷 Gita Guide")
st.caption("Ask a question. Answers are generated from your Bhagavad Gita PDF using RAG + Groq.")

if not config.GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Copy `.env.example` to `.env`, paste your key, and restart.")
    st.stop()

rag = get_rag()
if not rag.is_ready():
    st.warning("The vector database is empty. Put your PDF in the `data` folder and run `python ingest.py` first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve", 2, 10, config.TOP_K)
    show_sources = st.checkbox("Show retrieved passages", value=True)

    try:
        models = rag.list_chat_models()
    except Exception as e:
        st.error(f"Could not fetch models: {e}")
        st.stop()
    preferred = [m for m in models if "gpt-oss" in m or "llama" in m]
    default = config.GROQ_MODEL if config.GROQ_MODEL in models else (preferred or models)[0]
    model = st.selectbox("Groq model", models, index=models.index(default))

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(passages):
    with st.expander(f"Sources ({len(passages)} passages)"):
        for p in passages:
            st.markdown(f"**Page {p['page']}** · similarity {p['score']:.2f}")
            st.write(p["text"])
            st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("e.g. How do I deal with fear of failure?")
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            passages = rag.retrieve(question, top_k)
            answer = st.write_stream(rag.stream_answer(question, passages, history, model))
            if show_sources:
                render_sources(passages)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": passages}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")