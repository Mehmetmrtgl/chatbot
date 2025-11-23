# Chatbot Platform

A Flask-based chatbot platform that combines retrieval augmented generation (RAG), feedback collection, and lightweight admin tools. The backend stores question-answer pairs in PostgreSQL, retrieves related context with FAISS embeddings, and generates answers through an Ollama-hosted model.

## Architecture
- **Flask API** (`app/app.py`) with blueprints for chat, feedback, autocomplete, admin, chat history, and suggested questions.
- **PostgreSQL** schema managed through SQLAlchemy models (`app/db/models.py`) and helper scripts (`scripts/create_db.py`, `scripts/db_utils.py`).
- **Vector search (FAISS)** loaded from `data/embeddings/faiss_index` via LangChain (`scripts/rag_utils.py`) and autocomplete utilities (`scripts/faiss_utils.py`).
- **LLM inference** routed to an Ollama server using a Qwen model (`scripts/model_inference.py`). Generated answers can be stored for later review.
- **Client assets**: minimal HTML/JS in `app/templates/chat.html` and `app/static/chat.js` for manual testing, plus React dependencies listed in `package.json` for richer front-end work.

## Prerequisites
- Python 3.10+ and Node.js (if you plan to build a React UI)
- PostgreSQL running locally (default URI: `postgresql://postgres:123456@localhost:5432/hu_chatbot2`)
- An Ollama server listening on `http://localhost:11434/v1` with the `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0` model pulled or aliased
- FAISS index files at `data/embeddings/faiss_index` (see **Data preparation**)

## Setup
1. Create a virtual environment and install backend dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure PostgreSQL access. Update `app/app.py` or set the `SQLALCHEMY_DATABASE_URI` environment variable before launching (defaults to `postgresql://postgres:123456@localhost:5432/hu_chatbot2`).
3. Initialize the database tables:
   ```bash
   python -m scripts.create_db
   ```
4. Prepare vector data (see below) so RAG and autocomplete can run.
5. Start the Flask app:
   ```bash
   python -m app.app
   ```
   The development server binds to `http://0.0.0.0:5000`.

## Data preparation
- **Embeddings / RAG context**: Build a FAISS index into `data/embeddings/faiss_index` using scripts such as `scripts/create_faiss_from_pdf.py`, `scripts/generate_questions_from_pdfs.py`, or `scripts/augment_faiss_with_paraphrases.py` depending on your source documents.
- **Autocomplete index**: Use `scripts/create_autocomplete_index.py` to generate question embeddings for fast prefix matching.
- **Manual Q&A loading**: Import curated pairs with `scripts/load_manual_questions.py` or `scripts/load_reference_answers.py` before fine-tuning or review.
- **Fine-tuning dataset export**: Admin endpoints can write approved answers to `fine_tuning_data.jsonl`; the file path defaults to the repository root.

## Key API endpoints
- `POST /chat` — Main chat endpoint; logs user/bot messages, runs DB lookup, RAG retrieval, and Ollama generation.
- `POST /api/autocomplete` — Returns up to 5 similar questions for a given prefix.
- `POST /api/suggested_questions` — Combines FAISS-based similar questions with random prompts from `data/finetune/so.jsonl`.
- `POST /api/feedback` — Records like/dislike feedback linked to a `question_id`.
- `GET /api/chat_sessions` and `GET /api/chat_history/<session_id>` — Fetch stored chat logs per session.
- Admin utilities (prefixed with `/api/`): review unapproved answers, approve/update/reject answers, export fine-tuning data, fetch analytics, generate alternative answers, and retrieve Q&A lists (`app/routes/admin_routes.py`).

## Development tips
- **Chat logic**: `app/chat/utils.py` normalizes input, checks curated answers, falls back to FAISS context, evaluates answer quality, and logs messages.
- **Model selection**: Update `MODEL_MAP` in `scripts/model_inference.py` if you swap Ollama models.
- **Quality controls**: `scripts/quality_utils.py`, `scripts/quality_decision.py`, and `scripts/evaluate_answer_quality_llm.py` help benchmark generated answers and mark low-quality responses.
- **Testing the UI**: Open `app/templates/chat.html` in a browser while the server runs to submit questions via the lightweight JS client.

## Running React (optional)
React dependencies are declared in `package.json`. If you build a richer front end:
1. Initialize the client (e.g., in a `client/` directory) with your preferred React tooling.
2. Install packages with `npm install`.
3. Point API calls to the Flask server (default `http://localhost:5000`).

## Troubleshooting
- Ensure PostgreSQL extensions that provide `similarity` (e.g., `pg_trgm`) are available if fuzzy matching fails in `scripts/db_utils.get_answer_from_db`.
- Verify `data/embeddings/faiss_index` exists and is readable; otherwise `/chat` will fall back to suggestions without RAG context.
- If Ollama is unreachable, `/chat` requests will raise server errors—test connectivity with the script footer in `scripts/model_inference.py`.
