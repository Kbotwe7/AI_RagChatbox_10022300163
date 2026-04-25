"""
Streamlit UI — RAG assistant with optional full-page background image.
Override identity via .env: STUDENT_NAME, STUDENT_INDEX.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSET_IMAGES = ROOT / "assets" / "images"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from openai import APIStatusError, AuthenticationError, RateLimitError

from src.config import (
    BUDGET_PDF_URL,
    CSV_URL,
    DATA_PROCESSED,
    EMBEDDING_MODEL_NAME,
    GROK_MODEL,
    PROJECT_ROOT,
    TOP_K,
    XAI_API_KEY,
    student_header,
)
from src.feedback_store import FeedbackStore
from src.index_loader import load_retriever
from src.rag_pipeline import RagPipeline


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


@st.cache_data(show_spinner=False)
def _logo_data_url(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    if path.stat().st_size > 2 * 1024 * 1024:
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        ext = path.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
            ext, "image/png"
        )
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


MAX_BACKGROUND_BYTES = 20 * 1024 * 1024  # large base64 URLs can slow the browser; compress if bigger


@st.cache_data(show_spinner=False)
def _background_data_url(path_str: str, mtime_ns: int) -> str | None:
    """Embed local image for CSS background (Streamlit cannot load file:// URLs)."""
    path = Path(path_str)
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_BACKGROUND_BYTES:
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        ext = path.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
            ext, "image/png"
        )
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _inject_theme_css(bg_data_url: str | None) -> None:
    # Top layer first: very light wash so the campus photo stays sharp and clear.
    if bg_data_url:
        bg_layer = (
            "linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.06) 50%, rgba(248,250,252,0.05) 100%), "
            f'url("{bg_data_url}")'
        )
    else:
        bg_layer = "linear-gradient(145deg, #e2e8f0 0%, #cbd5e1 40%, #94a3b8 100%)"

    st.markdown(
        f"""
        <style>
          :root {{
            --ac-red: #b91c1c;
            --ac-red-soft: rgba(185, 28, 28, 0.12);
          }}
          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&display=swap');
          html, body, [class*="css"] {{ font-family: "DM Sans", system-ui, sans-serif; }}
          [data-testid="stAppViewContainer"] {{
            background-image: {bg_layer};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
          }}
          [data-testid="stHeader"] {{
            background: rgba(255,255,255,0.55);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid rgba(15,23,42,0.12);
          }}
          [data-testid="stToolbar"] {{ display: block; }}
          /* Make top toolbar/deploy controls visible on light header */
          [data-testid="stToolbar"] button,
          [data-testid="stToolbar"] a {{
            color: #0f172a !important;
            background: rgba(255,255,255,0.92) !important;
            border: 1px solid rgba(15,23,42,0.2) !important;
            border-radius: 8px !important;
          }}
          [data-testid="stToolbar"] button svg,
          [data-testid="stToolbar"] a svg {{
            fill: #0f172a !important;
            stroke: #0f172a !important;
          }}
          [data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.78) !important;
            backdrop-filter: blur(12px);
            border-right: 3px solid var(--ac-red);
            box-shadow: -4px 0 24px rgba(185, 28, 28, 0.06) inset;
          }}
          [data-testid="stSidebar"] .stMarkdown p,
          [data-testid="stSidebar"] .stMarkdown li,
          [data-testid="stSidebar"] .stMarkdown strong,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] span,
          [data-testid="stSidebar"] div,
          [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3,
          [data-testid="stSidebar"] h4,
          [data-testid="stSidebar"] h5,
          [data-testid="stSidebar"] h6 {{
            color: #000 !important;
          }}
          /* Make sidebar selectbox text/options clearly visible */
          [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid rgba(185,28,28,0.35) !important;
          }}
          [data-testid="stSidebar"] [data-baseweb="select"] span,
          [data-testid="stSidebar"] [data-baseweb="select"] input,
          [data-testid="stSidebar"] [data-baseweb="select"] div {{
            color: #000000 !important;
          }}
          div[data-baseweb="popover"] ul,
          div[data-baseweb="popover"] li,
          div[data-baseweb="popover"] span,
          div[data-baseweb="popover"] div {{
            background: #ffffff !important;
            color: #000000 !important;
          }}
          .block-container {{
            padding: 2rem 1.75rem 3rem 1.75rem;
            max-width: 920px;
            background: rgba(255,255,255,0.78);
            border-radius: 20px;
            border: 1px solid rgba(185, 28, 28, 0.18);
            box-shadow: 0 25px 50px -12px rgba(15,23,42,0.14);
            margin-top: 0.5rem;
          }}
          div[data-testid="stAlert"] {{ border-radius: 10px !important; }}
          .app-header {{
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--ac-red-soft);
          }}
          .app-header-row {{
            display: flex;
            align-items: flex-start;
            gap: 1.15rem;
            flex-wrap: wrap;
          }}
          .app-header .header-logo {{
            width: 76px;
            height: auto;
            max-height: 88px;
            object-fit: contain;
            flex-shrink: 0;
            margin-top: 0.15rem;
          }}
          .app-header .title-wrap {{
            border-left: 5px solid var(--ac-red);
            padding-left: 1rem;
            margin: 0;
            flex: 1;
            min-width: 200px;
          }}
          .app-header .title {{
            font-size: clamp(1.85rem, 3.2vw, 2.35rem);
            font-weight: 800;
            line-height: 1.15;
            margin: 0;
            letter-spacing: -0.03em;
          }}
          .app-header .title .ac {{ color: var(--ac-red); }}
          .app-header .title .rag {{ color: #0f172a; font-weight: 800; }}
          .app-header .subtitle {{ color: #475569; font-size: 0.95rem; margin: 0.35rem 0 0 0; line-height: 1.5; }}
          .app-header .meta {{
            margin-top: 0.75rem;
            font-size: 0.82rem;
            color: #64748b;
          }}
          .section-label {{
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #b91c1c;
            margin: 1.25rem 0 0.5rem 0;
            padding-left: 0.6rem;
            border-left: 3px solid var(--ac-red);
          }}
          /* Force readable dark text for chatbot answer/output area */
          .block-container [data-testid="stMarkdownContainer"] p,
          .block-container [data-testid="stMarkdownContainer"] li,
          .block-container [data-testid="stMarkdownContainer"] span,
          .block-container [data-testid="stMarkdownContainer"] strong {{
            color: #000 !important;
          }}
          div[data-testid="stExpander"] {{
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(203,213,225,0.95);
            border-radius: 12px;
          }}
          div[data-testid="stExpander"] summary {{
            color: #0f172a;
            border-radius: 8px;
            transition: background-color 0.15s ease, color 0.15s ease;
          }}
          div[data-testid="stExpander"] details[open] > summary {{
            background-color: #b91c1c !important;
            color: #fff !important;
          }}
          div[data-testid="stExpander"] summary:hover {{
            background-color: #b91c1c !important;
            color: #fff !important;
          }}
          .stTextArea textarea {{
            background: #fff !important;
            color: #0f172a !important;
            caret-color: #0f172a !important;
            border-radius: 12px !important;
            border: 1px solid rgba(185,28,28,0.22) !important;
          }}
          .stTextArea textarea:focus {{
            border-color: rgba(185,28,28,0.45) !important;
            box-shadow: 0 0 0 1px rgba(185,28,28,0.15) !important;
          }}
          .stButton > button {{
            background: #b91c1c !important;
            color: #fff !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.5rem 1.25rem !important;
            border-left: 4px solid #7f1d1d !important;
          }}
          .stButton > button:hover {{
            background: #991b1b !important;
            color: #fff !important;
            border-left-color: #ef4444 !important;
          }}
          h1, h2, h3 {{ color: #0f172a !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _resources():
    store, retriever, embedder = load_retriever()
    fb_path = DATA_PROCESSED / "feedback_weights.json"
    feedback = FeedbackStore(fb_path)
    pipeline = RagPipeline(retriever, embedder, feedback)
    return store, pipeline, feedback


def main() -> None:
    st.set_page_config(
        page_title="Academic City RAG System chatbot",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    bg_path = _first_existing(
        [
            ASSET_IMAGES / "background3.png",
            ASSET_IMAGES / "background3.jpg",
            ASSET_IMAGES / "background3.jpeg",
            ASSET_IMAGES / "background3.webp",
            ASSET_IMAGES / "background2.png",
            ASSET_IMAGES / "background2.jpg",
            ASSET_IMAGES / "background2.jpeg",
            ASSET_IMAGES / "background2.webp",
            ASSET_IMAGES / "backgr.png",
            ASSET_IMAGES / "backgr.jpg",
            ASSET_IMAGES / "background.png",
            ASSET_IMAGES / "background.jpg",
            ASSET_IMAGES / "bg.png",
            ASSET_IMAGES / "bg.jpg",
        ]
    )
    bg_url: str | None = None
    if bg_path:
        sz = bg_path.stat().st_size
        mtime_ns = int(bg_path.stat().st_mtime_ns)
        if sz > MAX_BACKGROUND_BYTES:
            st.warning(
                f"`{bg_path.name}` is about **{sz / (1024 * 1024):.1f} MB**; the app embeds up to **{MAX_BACKGROUND_BYTES // (1024 * 1024)} MB**. "
                "Export a smaller JPEG/PNG (e.g. 1920px wide) or the background will not load."
            )
        else:
            bg_url = _background_data_url(str(bg_path), mtime_ns)
    _inject_theme_css(bg_url)

    logo_path = _first_existing(
        [
            ASSET_IMAGES / "logo.png",
            ASSET_IMAGES / "logo.jpg",
            ASSET_IMAGES / "logo.webp",
            ASSET_IMAGES / "ac-logo.png",
            ASSET_IMAGES / "ac-logo.jpg",
        ]
    )
    logo_url = _logo_data_url(str(logo_path)) if logo_path else None
    logo_html = (
        f'<img class="header-logo" src="{logo_url}" alt="Logo" />'
        if logo_url
        else ""
    )

    meta = student_header()
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-header-row">
            {logo_html}
            <div class="title-wrap">
              <p class="title">
                <span class="ac">Academic City</span><span class="rag"> RAG System chatbot</span>
              </p>
            </div>
          </div>
          <p class="subtitle">Query the indexed Ghana election data and 2025 national budget. Each run shows retrieval, the prompt sent to the model, and the answer.</p>
          <p class="meta">{meta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bg_path is None:
        st.caption(
            "Tip: add a full-page background as **`background3.png`** or **`background3.jpg`** "
            f"in `{ASSET_IMAGES}` (under ~10MB). Optional: `feature.png` / `.jpg` for a second column."
        )

    feature_path = _first_existing(
        [
            ASSET_IMAGES / "feature.png",
            ASSET_IMAGES / "feature.jpg",
        ]
    )

    try:
        _, pipeline, feedback = _resources()
    except Exception as e:
        st.error(
            "Index not found or failed to load. Run `python scripts/build_index.py` from the project root first."
        )
        st.exception(e)
        return

    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        st.info(
            f"No `.env` file at `{env_path}`. Copy `.env.example` to `.env`, set keys for your provider, "
            "then restart Streamlit."
        )
    elif not (XAI_API_KEY or "").strip():
        st.info(
            f"`XAI_API_KEY` (or `GROK_API_KEY`) is empty in `{env_path}`. Create a key at "
            "[console.x.ai](https://console.x.ai/), add it to `.env`, then restart Streamlit."
        )
    elif "your-key-here" in (XAI_API_KEY or "").lower():
        st.info(
            "`XAI_API_KEY` in `.env` is still a **placeholder**. Replace it with your real key from "
            "[console.x.ai](https://console.x.ai/), save, and restart Streamlit."
        )

    with st.sidebar:
        with st.expander("What this app does", expanded=False):
            st.markdown(
                f"""
**Purpose**  
This is a **RAG** (retrieval-augmented generation) demo: your question is used to **search** pre-loaded text chunks; the best-matching chunks are **injected into a prompt**; an **LLM** writes the final answer so it can stay grounded in those chunks.

**Where answers come from (knowledge base)**  
Only text that was **ingested** when you ran `scripts/build_index.py`:

1. **Ghana election results (CSV)** — downloaded from the course dataset repo.  
   `{CSV_URL}`

2. **Ghana 2025 Budget Statement (PDF)** — Ministry of Finance.  
   `{BUDGET_PDF_URL}`

Those files are **cleaned, split into chunks, embedded**, and stored locally under `data/processed/` (FAISS + keyword index). The app **does not** browse the live web at question time.

**Models**  
- **Embeddings (retrieval):** `{EMBEDDING_MODEL_NAME}` (local, via Sentence-Transformers).  
- **Chat (answers):** **xAI Grok** via `XAI_API_KEY` (or `GROK_API_KEY`) and optional `GROK_MODEL` (default **`{GROK_MODEL}`**).  
  Keys: [console.x.ai](https://console.x.ai/) — docs: [xAI API](https://docs.x.ai/).

**Without RAG**  
If you turn off “Use RAG”, Grok answers **only** from its general training data, **not** from your CSV/PDF chunks.
                """.strip()
            )
        st.markdown("**Controls**")
        use_rag = st.toggle("Use RAG (retrieval)", value=True)
        use_hybrid = st.toggle("Hybrid search (BM25 + vector)", value=True)
        use_qexp = st.toggle("Query expansion (light)", value=False)
        top_k = st.slider("Top-k", 3, 20, TOP_K)
        variant = st.selectbox("Prompt variant", ["strict", "concise", "exploratory"])
        use_feedback = st.toggle("Apply feedback re-rank", value=True)

    if feature_path:
        col_chat, col_img = st.columns((1.35, 1), gap="large")
    else:
        col_chat = st.container()
        col_img = None

    with col_chat:
        st.markdown('<p class="section-label">Ask</p>', unsafe_allow_html=True)
        query = st.text_area("Question", height=110, label_visibility="collapsed", placeholder="e.g. What macroeconomic targets are stated for 2025?")
        run = st.button("Run")

        if run and query.strip():
            try:
                with st.spinner("Running pipeline..."):
                    trace = pipeline.run(
                        query.strip(),
                        use_rag=use_rag,
                        use_hybrid=use_hybrid,
                        use_query_expansion=use_qexp,
                        prompt_variant=variant,
                        apply_feedback=use_feedback,
                        top_k=top_k,
                    )
            except AuthenticationError:
                st.error(
                    "**401 — invalid API key** from xAI (Grok). Check **`XAI_API_KEY`** (or **`GROK_API_KEY`**) in "
                    f"`{PROJECT_ROOT / '.env'}` — no quotes, no spaces around `=`. "
                    "Create or rotate a key at [console.x.ai](https://console.x.ai/), save, restart Streamlit."
                )
                return
            except RateLimitError:
                st.error(
                    "**429 — rate limit or quota** from xAI. Wait and retry, or check your plan and limits in "
                    "[console.x.ai](https://console.x.ai/). You can try another model id via **`GROK_MODEL`** in `.env`."
                )
                return
            except APIStatusError as api_err:
                if getattr(api_err, "status_code", None) == 429:
                    st.error(
                        "**429** from xAI. See the error message in your terminal; often this is rate limits or "
                        "account quota — [console.x.ai](https://console.x.ai/)."
                    )
                    return
                raise
            except RuntimeError as err:
                if "XAI_API_KEY" in str(err) or "GROK_API_KEY" in str(err):
                    st.error(
                        f"{err} Restart Streamlit after saving `.env` in the project folder: `{PROJECT_ROOT}`."
                    )
                else:
                    st.exception(err)
                return
            except Exception as err:
                st.exception(err)
                return

            st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
            st.write(trace.get("final_answer", ""))
            st.caption(f"Latency: {trace.get('latency_s', 0):.2f}s")

            for stage in trace.get("stages", []):
                name = stage.get("name", "stage")
                with st.expander(f"Stage: {name}", expanded=name in ("retrieval", "prompt")):
                    if name in ("retrieval", "feedback_rerank"):
                        hits = stage.get("hits", [])
                        for h in hits:
                            cols = st.columns([6, 1, 1])
                            cols[0].markdown(
                                f"**#{h.get('rank', 0)}** score=`{h.get('score', 0):.4f}` "
                                f"chunk_id=`{h.get('chunk_id')}`"
                            )
                            cols[0].markdown(f"> {h.get('text', '')[:1200]}")
                            if cols[1].button("👍", key=f"up-{name}-{h.get('chunk_id')}"):
                                feedback.record(int(h["chunk_id"]), "up")
                                st.success("Recorded")
                            if cols[2].button("👎", key=f"down-{name}-{h.get('chunk_id')}"):
                                feedback.record(int(h["chunk_id"]), "down")
                                st.success("Recorded")
                    elif name == "prompt":
                        st.code(json.dumps(stage.get("messages", []), indent=2), language="json")
                    else:
                        st.json(stage)

    if col_img is not None and feature_path:
        with col_img:
            st.markdown('<p class="section-label">Figure</p>', unsafe_allow_html=True)
            st.image(str(feature_path), use_container_width=True)


if __name__ == "__main__":
    main()
