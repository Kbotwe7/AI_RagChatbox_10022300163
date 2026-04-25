# Demo video narration (~3–5 minutes)

Speak in a calm, clear tone. Pause where noted. Keep the browser tab with the app visible for screen capture.

---

## 1. Opening (0:00–0:25)

“Hi, I’m **[your name]**, index **[your index]**. This is my **RAG assistant** for **Ghana election results** and the **2025 national budget**. I built retrieval and prompting myself—no LangChain or LlamaIndex—and I use **Sentence-Transformers** for embeddings, **FAISS** and **BM25** for search, and **xAI Grok** to write the final answer from the retrieved context.”

*[Pause. Show the app header and background.]*

---

## 2. What the app does (0:25–0:50)

“On screen is the **Streamlit** interface. The sidebar has toggles for **RAG**, **hybrid search**, **query expansion**, and **top‑k**. After each answer I can open **stages** to see **retrieval hits**, the **prompt** sent to the model, and timing.”

*[Open the sidebar briefly, then collapse or leave as you prefer.]*

---

## 3. Live question — election (0:50–2:00)

“I’ll ask a question that should be answered **only from the indexed CSV**, for example: *What percentage of votes did John Dramani Mahama get in 2020 in Savannah Region?*”

*[Type or paste the question. Click **Run**. Wait for the answer.]*

“Here the model is **grounded** in the chunks we retrieved—you can expand **Stage: retrieval** to see scores and snippet text, and **Stage: prompt** to see how context was injected.”

---

## 4. Optional — budget or second question (2:00–2:45)

“For a **budget** question, the same pipeline pulls from **PDF chunks**, for example macroeconomic targets or revenue measures for 2025.”

*[Optional: run one short budget query if time.]*

---

## 5. Innovation — feedback (2:45–3:15)

“For the coursework **innovation**, I store **thumbs up/down** on chunks and use that to **re-rank** retrieval on later runs, so repeated feedback nudges the system toward better sources.”

*[Expand a retrieval stage; click a thumb if your recording allows.]*

---

## 6. Closing (3:15–end)

“That’s the **Academic City RAG chatbot**: custom indexing, hybrid retrieval, grounded prompts with **Grok**, and transparent stages for grading and debugging. Thanks for watching.”

---

## Quick reference — sample queries

- Election: `What percentage of votes did John Dramani Mahama get in 2020 in Savannah Region?`
- Election: `Who was victorious in Savannah Region in 2020?`
- Budget (example): `What macroeconomic targets are stated for 2025?`

---

## Streamlit Community Cloud checklist

1. Commit and push: `git push origin main` (or your deploy branch).
2. On [share.streamlit.io](https://share.streamlit.io), **New app** → pick **this GitHub repo** and branch → main file **`app.py`**.
3. Add **Secrets** in the app settings for `XAI_API_KEY` (and any other vars from `.env.example`).
4. Cloud has **no local index** by default—either commit `data/processed/` if allowed, or document that the grader runs locally / you upload artifacts per course instructions.
