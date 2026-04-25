# Retrieval-Augmented Generation (RAG) AI Assistant for Ghana Election and Budget Analysis

## Student Details

- **Name:** Kojo Baafi Botwe  
- **Index Number:** 10022300163  
- **Lecturer:**  Dr. Godwin N. Danso  
- **Submission email:** godwin.danso@acity.edu.gh  
- **Email subject:** `Introduction to Artificial Intelligence-2026:[10022300163 Kojo Baafi Botwe]`  
- **GitHub collaborator:** `godwin.danso@acity.edu.gh` / `GodwinDansoAcity`


## Project Overview

This project presents a manually implemented **Retrieval-Augmented Generation (RAG)** assistant that answers user questions from:

- **Structured data:** Ghana election results (CSV)  
- **Unstructured data:** Ghana 2025 Budget Statement (PDF)

The goal is to deliver accurate, context-grounded answers while reducing hallucination, without using orchestration frameworks such as LangChain/LlamaIndex.

## What this project is

A **manual RAG** chat assistant with custom chunking, embeddings (Sentence-Transformers), FAISS vector search, BM25 + dense **hybrid retrieval**, optional **query expansion**, prompt construction with **context limits** and hallucination controls, **xAI Grok** generation (OpenAI-compatible client), and stage logging visible in the UI.

**Innovation (Part G):** persisted **user feedback** (thumbs up/down per chunk) adjusts retrieval scores before context selection.



## Report-Aligned Highlights (Rubric)

- **Part A (Data engineering & chunking):** CSV/PDF preprocessing + chunking design and comparison.  
- **Part B (Custom retrieval):** embeddings + FAISS + BM25 hybrid + optional query expansion.  
- **Part C (Prompting & generation):** strict grounding prompts with fallback for missing evidence.  
- **Part D (Full pipeline):** query -> retrieval -> re-rank -> context selection -> prompt -> generation with stage logs.  
- **Part E (Evaluation):** factual, ambiguous, unsupported, and paraphrased query testing + RAG vs non-RAG comparison.  
- **Part F (Architecture):** modular design documented in `docs/ARCHITECTURE.md`.  
- **Part G (Innovation):** feedback-based retrieval re-ranking.

## Data sources

- CSV: [Ghana_Election_Result.csv](https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv) (downloaded at build time).  
- PDF: [2025 Budget Statement (v4)](https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf) — the exam handout merged a line break into `andEconomic`; the live file uses `and-Economic`. The ingest script tries the correct URL first, then fallbacks.
- Extra local datasets: place additional `.csv` or `.pdf` files in `data/raw/`; `scripts/build_index.py` auto-loads them into the same index (besides the main election CSV and budget PDF).

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add `XAI_API_KEY` (Grok / xAI), `STUDENT_NAME`, and `STUDENT_INDEX` to `.env`.

## Build the index

```bash
python scripts/build_index.py
```

## Run the app

```bash
streamlit run app.py
```

### Streamlit Community Cloud (missing index)

Cloud clones do not include `data/processed/` (it is gitignored). Either:

1. **Auto-build on the server (simplest):** In the deployed app go to **Manage app → Settings → Secrets** and add:
   - `XAI_API_KEY` = your key (required for answers)
   - `RAG_BUILD_INDEX_IF_MISSING` = `1`  
   Save, **Reboot** the app, then open it once and wait several minutes while the first run downloads the CSV/PDF and builds the FAISS index.

2. **Ship a pre-built index:** Run `python scripts/build_index.py` locally, then commit `data/processed/index_store/` and adjust `.gitignore` if your course allows large binaries in Git.

---

## Evaluation Queries Used

1. What percentage of votes did John Dramani Mahama get in 2020 in Savannah Region?  
2. Who won the election in Ketu North?  
3. Who won the election?  
4. Who was victorious in Savannah Region in 2020?

Expected behavior:
- Accurate factual answer when context is present
- `I don't know` / refusal for unsupported or ambiguous cases
- Correct handling of paraphrased winner queries through query expansion + semantic retrieval

## Part A — Chunking justification (summary)

- **Character window 450 / overlap 90 (~20%)** balances (1) enough local context per chunk for the embedding model, (2) boundary continuity so entities split across windows still appear in at least one chunk, (3) UI/prompt budget.  
- **Comparative analysis:** run `python scripts/compare_chunking.py` after the index build (uses election CSV) to compare a smaller vs larger overlap configuration on chunk count, mean length, and top-3 dense retrieval scores for fixed queries. Record observations manually in `experiments/MANUAL_EXPERIMENT_LOG.md`.

## Part B — Failure cases and fix

- **Failure:** short acronyms (e.g. “GDP”) may retrieve semantically related but not budget-explicit passages under dense-only search.  
- **Fix implemented:** hybrid BM25 + vector fusion; optional **query expansion** maps acronyms to spelled-out domain phrases (`src/retrieval.py`).

## Parts C–E

- Prompt variants (`strict` / `concise` / `exploratory`) in `src/prompts.py` — compare outputs for the same query in the UI and log differences in the manual experiment file.  
- **RAG vs pure LLM:** toggle “Use RAG” off in the sidebar to run the same query without retrieval; log factual differences against the CSV/PDF ground context manually.

## UI stage guide

When you run a query, the app logs these stages in the answer section:

- `retrieval`: top-k chunks fetched from FAISS/BM25 with similarity/fusion scores.
- `feedback_rerank`: re-ordered hits after applying user thumbs up/down weights.
- `context_selection`: final chunks selected under the context-size budget.
- `prompt`: exact messages sent to Grok (system + user with injected context).
- `generation`: model output text and model id used for that response.



