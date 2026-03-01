# FCAI RAG API Documentation

> **Repository**: Retrieval-Augmented-Generation (RAG) API for the Faculty of Computer & Artificial Intelligence Regulations (PDF)

---

## Overview

This Flask-based API exposes a small Retrieval-Augmented-Generation (RAG) service built with LangChain components, a Chroma vector store, multilingual embeddings (HuggingFace `intfloat/multilingual-e5-base`), and Google Generative AI (Gemini) as the LLM. It is designed to answer questions about a specific PDF (the FCAI internal regulations PDF) in Arabic or English and return source snippets/pages.

Key features:
- Loads and preprocesses a PDF with Arabic reshaping and BiDi reordering for correct display.
- Splits content to chunks for vector indexing.
- Builds or loads a persistent Chroma vector store.
- Uses Hugging Face multilingual embeddings for vectorization.
- Uses Google Generative AI (`gemini-2.5-flash`) as the LLM for answers.
- Exposes simple HTTP endpoints for health checks and chat queries.

---

## File: `app.py` (summary)

The main application file performs the following responsibilities:

1. Loads the PDF using `PyPDFLoader`.
2. Applies `clean_arabic_text(...)` to each page to reshape and reorder Arabic.
3. Splits the text into chunks with `RecursiveCharacterTextSplitter`.
4. Creates / loads a Chroma vectorstore using `HuggingFaceEmbeddings`.
5. Builds a `RetrievalQA` chain wrapping the Google Generative AI model.
6. Exposes two endpoints:
   - `GET /` — health check
   - `POST /api/chat` — accepts `{"query": "..."}` and returns the answer plus sources
7. Initializes the RAG chain at import time (so both `python app.py` and Gunicorn imports run `initialize_rag_chain()`).

---

## Requirements

Create a `requirements.txt` with (example):

```
flask
langchain
langchain-community
langchain-core
langchain-google-genai
langchain-huggingface
chromadb
arabic-reshaper
python-bidi

# pin versions as needed for reproducibility
```

> Note: Some package names differ across PyPI / package forks; validate names and versions in your environment and pin versions for production.

---

## Environment Variables

The application expects the following environment variables:

- `GOOGLE_API_KEY` — **required**. API key for Google Generative AI (Gemini). If missing, the server will log an error and the `/api/chat` endpoint will return HTTP 500.
- `PORT` — optional. Port to run the Flask dev server (default: `7860`).

Set these in your environment or via a `.env` file (when using a process manager that supports it) or in the Docker image.

---

## Configuration Constants (in `app.py`)

- `pdf_path` — Path to the PDF to index. Default: `./USC Faculty of Computer and Artificial Intelligence Internal Regulations (October 2019).pdf`.
- `persist_directory` — Directory for Chroma persistence. Default: `./chroma_fcai_regulations_db`.
- `chunk_size` / `chunk_overlap` — Configured in the text splitter (currently: 3000 / 1000). Tune these if memory or retrieval relevance needs adjustments.

---

## Endpoints

### `GET /`
**Description:** Simple health check.

**Response (200):**

```json
{ "status": "API is running and healthy" }
```

---

### `POST /api/chat`
**Description:** Main chat endpoint. Runs the RetrievalQA chain with the provided question and returns a text answer plus a list of source documents (with page numbers).

**Request body (JSON):**

```json
{ "query": "How many credits are required for graduation?" }
```

**Success response (200):**

```json
{
  "answer": "...full text answer...\n\n📚 المصادر:\n  • <source-path-or-id> | صفحة <page-number>\n  • ..."
}
```

**Error responses:**
- `400` — when `query` is missing in the request body.
- `500` — when `GOOGLE_API_KEY` is not configured or the RAG chain is uninitialized or chain invocation fails.

---

## How it works (detailed)

1. **Document loading & Arabic cleaning**
   - `PyPDFLoader` reads the PDF into a list of page `Document` objects.
   - `clean_arabic_text()` reshapes Arabic text using `arabic_reshaper.reshape()` and reorders characters with `bidi.algorithm.get_display()` so right-to-left Arabic displays correctly when later shown.

2. **Splitting**
   - `RecursiveCharacterTextSplitter` creates overlapping chunks to preserve context across chunk boundaries. Current separators: `\n\n`, `\n`, `.`, and space.

3. **Embeddings & Vectorstore**
   - `HuggingFaceEmbeddings` using `intfloat/multilingual-e5-base` encodes all chunks.
   - If `persist_directory` exists, the app loads the Chroma vectorstore from disk to avoid re-indexing.
   - If missing, it creates the vectorstore from chunks and calls `.persist()`.

4. **Retriever**
   - `vectorstore.as_retriever()` with `search_type='mmr'` and tuned `search_kwargs` (`k=8, fetch_k=25, lambda_mult=0.7`) is used to fetch relevant chunks for a query.

5. **LLM / Chain**
   - `ChatGoogleGenerativeAI` is instantiated (Gemini model) with `temperature=0` for deterministic answers.
   - A `ChatPromptTemplate` is used to inject the retrieved `context` and the `question` with clear instructions in both Arabic and English.
   - `RetrievalQA.from_chain_type(..., chain_type="stuff")` builds the final chain and is returned for use.

---

## Deployment Recommendations

### Development
- Run locally:
  ```bash
  export GOOGLE_API_KEY="your_key_here"
  python app.py
  ```
- Confirm `chroma_fcai_regulations_db` is created after first run (if not present).

### Production
- Use Gunicorn + workers (do not run Flask's dev server). Example Gunicorn command:
  ```bash
  gunicorn -w 2 -k gevent "app:app" --bind 0.0.0.0:7860
  ```
  - The app is initialized at import-time (the code calls `initialize_rag_chain()` in the `else` block), so the chain will be loaded once and reused by workers. Be mindful of memory usage.

- **Dockerfile** suggestions:
  - Base image: `python:3.11-slim` (or similar)
  - Install system deps required by `chromadb` or `sentence-transformers` if needed.
  - Copy code, install `requirements.txt` and set `ENV GOOGLE_API_KEY` in the runtime environment or via container orchestration.
  - Run with Gunicorn in CMD.

- **Kubernetes / Cloud**
  - Store `GOOGLE_API_KEY` securely in Secrets.
  - Use persistent volume for the Chroma `persist_directory` if you want vector store reuse across pod restarts.
  - Consider building the vectorstore in a separate job/container, write the persisted DB to a shared volume, and mount it into the API pods to reduce startup time.

---

## Performance & Scaling Notes

- **Startup time**: Building embeddings and persisting Chroma can be slow on the first run. Pre-build the vectorstore in CI/CD or a separate job and mount it.
- **Memory**: LLM clients may keep connections and large response buffers. Tune Gunicorn worker count based on available RAM.
- **GPU**: If you have GPU available, change `model_kwargs={'device': 'cuda'}` in `HuggingFaceEmbeddings` to speed up embedding generation (and ensure the base image has CUDA drivers and compatible PyTorch).
- **Retriever tuning**: Adjust `k`, `fetch_k`, and `lambda_mult` to change the tradeoff between diversity and relevance.

---

## Security

- Never commit `GOOGLE_API_KEY` to source control.
- Use HTTPS in front of the API in production (e.g., via ingress controller or load balancer).
- Apply rate limiting / authentication if exposing the endpoint publicly.

---

## Troubleshooting

- **`Error: PDF file not found`**: Verify `pdf_path` points to the correct location and the file exists in the container or host.
- **`Error: GOOGLE_API_KEY environment variable not set.`**: Set the env var. The `/api/chat` endpoint returns HTTP 500 when the key is absent.
- **Slow / timed out chain invocation**: Increase LLM timeouts or reduce chunk sizes; consider using a smaller embedding model or precomputing embeddings.
- **Incorrect Arabic rendering in responses**: The application reshapes input PDF text for correct indexing/display, but the LLM-generated answers may still require correct rendering in the client. Ensure the frontend supports RTL rendering (CSS `direction: rtl;`) and UTF-8.

---

## Example `curl` usage

```bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هي فترة امتحانات نهاية الفصل الدراسي؟"}'
```

Sample response body (abbreviated):

```json
{
  "answer": "يبدأ امتحان نهاية الفصل الدراسي ...\n\n📚 المصادر:\n  • ./.../USC Faculty ...pdf | صفحة 12"
}
```

---

## Testing Tips

- Add unit tests to verify:
  - `clean_arabic_text()` produces expected reshaped output for representative Arabic snippets.
  - `initialize_rag_chain()` returns a chain object when `GOOGLE_API_KEY` is set and the PDF exists.
  - `/api/chat` returns 400 when `query` is missing.

- Use Postman or HTTPie for interactive exploration.

---

## Future Improvements

- Add authentication (API keys / JWT) for the `/api/chat` endpoint.
- Support multiple PDFs and a management endpoint to (re)index documents.
- Add caching of LLM responses for repeated questions.
- Add a lightweight frontend (static HTML + JS) to test RTL rendering and present source links.
- Switch to a background task queue (Celery / RQ) for expensive re-indexing jobs.

---

## Contact / Maintainer

- **Maintainer:** Ahmed Ayman
- Keep `requirements.txt` pinned and document any manual system dependencies required by `chromadb` or `sentence-transformers`.

---

*End of document.*

