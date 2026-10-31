"""Retrieval + generation logic (used by app.py)."""
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

import config

SYSTEM_PROMPT = """You are a wise, gentle guide who answers questions using ONLY the \
passages from the Bhagavad Gita provided in the CONTEXT below.

Rules:
- Base your answer on the context. Do not invent verses, chapter numbers or quotes.
- If the context does not contain enough to answer, say so honestly and suggest how \
the user could rephrase the question.
- Explain in simple, clear, compassionate language. Translate or paraphrase any \
Sanskrit/ancient wording into plain modern language.
- Reply in the same language the user writes in.
- When you rely on a passage, mention its page like (p. 42).
- For serious personal, medical or mental-health struggles, remind the user to also \
talk to a trusted person or professional."""


class GitaRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        self.collection = client.get_or_create_collection(
            config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.llm = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None

    def is_ready(self) -> bool:
        return self.collection.count() > 0

    def list_chat_models(self) -> list[str]:
        """Ask Groq which models this API key can use right now."""
        skip = ("whisper", "tts", "guard", "embed", "orpheus", "playai")
        ids = [m.id for m in self.llm.models.list().data]
        return sorted(i for i in ids if not any(s in i.lower() for s in skip))

    def retrieve(self, question: str, k: int = config.TOP_K):
        emb = self.embedder.encode([question], normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": t, "page": m.get("page"), "source": m.get("source"), "score": 1 - d}
            for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def stream_answer(self, question: str, passages: list[dict], history: list[dict], model: str):
        context = "\n\n".join(
            f"[Passage {i} | page {p['page']}]\n{p['text']}" for i, p in enumerate(passages, 1)
        )
        messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
        messages += history[-6:]   # last few turns so follow-up questions work
        messages.append({"role": "user", "content": question})

        stream = self.llm.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta