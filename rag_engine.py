"""
DaxView RAG Engine — Groq (primary) + Gemini (fallback)
=========================================================
- Groq: ultra-fast inference (llama-3.3-70b-versatile)
- Gemini: fallback if Groq hits rate limit or fails
- Local ChromaDB embeddings (no API needed for vector search)
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

load_dotenv()

SYSTEM_PROMPT = """You are DaxView, an expert demand analyst for Daxwell — a single-use
disposables company serving healthcare, food service, and industrial customers.

You have two data sources:
1. VECTOR CONTEXT: EDA summaries, forecast narratives, anomaly reports
2. SQL RESULTS: live query results from the demand database

Guidelines:
- Be concise and specific. Cite numbers when available.
- Relate insights to healthcare, food service, and industrial verticals.
- Flag procurement risks or opportunities proactively.
- If data is insufficient, say so clearly.
"""


def _call_groq(prompt: str) -> str:
    """Call Groq API — llama-3.3-70b, ultra fast."""
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.3
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str) -> str:
    """Call Gemini API — fallback."""
    import google.generativeai as genai
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
    return model.generate_content(full_prompt).text


def _call_llm(prompt: str) -> tuple[str, str]:
    """Try Groq first, fall back to Gemini. Returns (response, provider)."""
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if groq_key:
        try:
            return _call_groq(prompt), "groq"
        except Exception as e:
            print(f"[Groq failed: {e}] — falling back to Gemini")

    if gemini_key:
        try:
            return _call_gemini(prompt), "gemini"
        except Exception as e:
            raise RuntimeError(f"Both Groq and Gemini failed. Last error: {e}")

    raise ValueError("No API keys found. Add GROQ_API_KEY or GEMINI_API_KEY to .env")


def _generate_sql_only(question: str, schema: str) -> str:
    """Generate SQL — tries Groq then Gemini, returns raw SQL."""
    prompt = f"""Generate a single SQLite query. Return ONLY SQL, no markdown, no backticks, no explanation.
Schema: {schema}
Rules: LIMIT 20 rows, use aggregations, Date column is TEXT in YYYY-MM-DD format.
Question: {question}"""

    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            return resp.choices[0].message.content.strip().replace("```sql","").replace("```","").strip()
        except Exception:
            pass

    if gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt).text.strip().replace("```sql","").replace("```","").strip()

    raise ValueError("No API keys available for SQL generation")


class DaxViewRAG:
    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not groq_key and not gemini_key:
            raise ValueError("Add GROQ_API_KEY and/or GEMINI_API_KEY to your .env file")

        self.db_path = Path("data/processed/demand.db")
        self._setup_vectorstore()

        # Show which providers are active
        providers = []
        if groq_key:
            providers.append("Groq (primary)")
        if gemini_key:
            providers.append("Gemini (fallback)")
        print(f"[DaxView] Active providers: {' → '.join(providers)}")

    def _setup_vectorstore(self):
        persist_dir = Path("vectorstore")
        persist_dir.mkdir(exist_ok=True)
        self.ef = DefaultEmbeddingFunction()
        self.chroma = chromadb.PersistentClient(path=str(persist_dir))
        try:
            self.collection = self.chroma.get_collection("daxview_local")
        except Exception:
            self.collection = self.chroma.create_collection(
                "daxview_local", embedding_function=self.ef
            )
            self._ingest_knowledge_docs()

    def _ingest_knowledge_docs(self):
        docs_dir = Path("knowledge_docs")
        if not docs_dir.exists():
            return
        documents, ids, metadatas = [], [], []
        for md_file in docs_dir.glob("*.md"):
            text = md_file.read_text()
            chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 50]
            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{md_file.stem}_{j}")
                metadatas.append({"source": md_file.stem})
        if documents:
            self.collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def _get_schema(self) -> str:
        if not self.db_path.exists():
            return "Database not found. Run 01_eda.py first."
        conn = sqlite3.connect(self.db_path)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        parts = []
        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
            parts.append(f"Table `{t}`: " + ", ".join(f"{c[1]} ({c[2]})" for c in cols))
        conn.close()
        return "\n".join(parts)

    def _run_sql(self, sql: str) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return df.to_string(index=False) if not df.empty else "No results."
        except Exception as e:
            return f"SQL error: {e}"

    def _vector_search(self, question: str, n: int = 4) -> tuple:
        try:
            results = self.collection.query(query_texts=[question], n_results=n)
            docs = results["documents"][0] if results["documents"] else []
            sources = list({m["source"] for m in results["metadatas"][0]}) if results["metadatas"] else []
            return "\n\n---\n\n".join(docs), sources
        except Exception as e:
            return f"Vector search unavailable: {e}", []

    def query(self, question: str, df: pd.DataFrame = None) -> dict:
        sources = []

        # 1. Vector search
        vector_context, vec_sources = self._vector_search(question)
        sources.extend(vec_sources)

        # 2. SQL agent
        sql_context = ""
        schema = self._get_schema()
        if "not found" not in schema:
            try:
                sql = _generate_sql_only(question, schema)
                result = self._run_sql(sql)
                sql_context = f"SQL: {sql}\n\nResults:\n{result}"
                sources.append("live_database")
            except Exception as e:
                sql_context = f"SQL generation failed: {e}"

        # 3. Dataframe stats
        df_context = ""
        if df is not None and not df.empty:
            top = df.groupby("Product_Category")["Order_Demand"].sum().sort_values(ascending=False).head(5)
            df_context = f"Top 5 categories:\n{top.to_string()}"

        # 4. Build prompt and call LLM
        prompt = f"""Question: {question}

VECTOR KNOWLEDGE:
{vector_context or 'None'}

SQL RESULTS:
{sql_context or 'Unavailable'}

DATAFRAME STATS:
{df_context or 'None'}

Answer concisely with specific numbers:"""

        answer, provider = _call_llm(prompt)
        sources.append(f"llm:{provider}")

        return {"answer": answer, "sources": sources, "provider": provider}
