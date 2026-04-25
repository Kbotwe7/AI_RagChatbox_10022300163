

**Student name:** Kojo Baafi Botwe 
**Index number:** 10022300163
**Instructions:** Append dated entries as you run tests.


**Model output (short excerpt):**  


**Compared to pure LLM (same query, RAG off):**  


## Chunking comparison (Part A)

Run `python scripts/compare_chunking.py` and paste **your** table / notes here.
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 




## Prompt A/B (Part C)
this is with strict
Ask Who was the loser in Savannah Region in 2020?


Answer I assume the question refers to the candidate with the fewest votes in the 2020 election in the Savannah Region. Based on the provided context, Bridget Dzogbenuku of the PPP received the lowest number of votes with 160, which is 0.07% of the total votes in the Savannah Region for the 2020 election (Context [8]). Therefore, Bridget Dzogbenuku can be considered the loser in terms of vote count among the candidates listed.


this is with concise:
Ask Who was the loser in Savannah Region in 2020?

Answer In the 2020 election in the Savannah Region, the candidate with the fewest votes was Bridget Dzogbenuku of the PPP, who received 160 votes, representing 0.07% of the total votes. This information is based on the data provided in the context (source [8]).
---

## Adversarial queries (Part E)
1. **Ambiguous:**
Ask who won elections
Answer I cannot answer from the provided documents.

Ask how many regions are there

Answer I can identify the following regions from the provided context: Central Region, Northern Region, Upper West Region, Greater Accra Region, and Brong Ahafo Region. This makes a total of 5 regions mentioned in the documents (based on entries [1], [2], [4], [6], and [7]). However, I am not sure if this represents the total number of regions in Ghana, as the context may not include all regions. If you are asking about the total number of regions in Ghana, I cannot answer definitively from the provided documents.
2. **Misleading / incomplete:**  
Ask how many regions are there(Rag)

Answer I can identify the following regions from the provided context: Central Region, Northern Region, Upper West Region, Greater Accra Region, and Brong Ahafo Region. This makes a total of 5 regions mentioned in the documents (based on entries [1], [2], [4], [6], and [7]). However, I am not sure if this represents the total number of regions in Ghana, as the context may not include all regions. If you are asking about the total number of regions in Ghana, I cannot answer definitively from the provided documents.

Ask how many regions are there (non rag)

Answer The number of regions depends on the context you're referring to. Could you please clarify what you're asking about? For example, are you asking about:
Geographical regions in a specific country or continent?
Administrative regions or states in a particular nation?
Regions in a game, app, or fictional world?
Biological or climatic regions?
Here are a few common contexts for clarity:
World Geographical Regions: The United Nations divides the world into 5 major regions (Africa, Americas, Asia, Europe, and Oceania), though sometimes these are further subdivided (e.g., North America, South America).
Regions in a Country: For example, the United States has 4 commonly recognized regions (Northeast, Midwest, South, and West) as defined by the U.S. Census Bureau.
Gaming Regions: Many games or platforms (like Pokémon or server regions in online games) have specific "regions" tied to gameplay or location.
Please provide more details, and I'd be happy to give a precise answer!



## Failure case log (Part B)
openai.AuthenticationError: Error code: 401 - (error: (message': 'Incorrect API key provided: sk-your-****here. You can find your API key at https://platform.openai.com/account/api-keys.,'type: invalid_requesterror, 'codel invalid_api_key param: None), status: 401)
Traceback:
File "C:\users\kojob\OneDrive\Desktop\AI_EOS|.venv\Lib\site-packages\streamlfE)runtime\scriptrunr
result F funco
File "C: \Users\kojob\OneDrive\Desktop\AI_EOS\.ven\Lib\site-packages\streamlit\runtime\scriptrunr
exec (code, module.__dict__) # noga: $102
٣٨٨٨٨٨٨٨٨٨٨٨
File "C:\Users\kojob|OneDrive\Desktop\AIEOS\app.py", line 436, in module›
ma in (
File "C: \users\kojob\OneDrive\Desktop\I_EOS\ app.py", Line 384, in matin
trace - pipeline.rund
query strip),
...<5 lines>...
top_k=top_k,
File "C:\Users\kojob\OneDrive\Desktop\AI_EOS\src\rag_pipeline.py", Line 115, in run
gen = chat complete(messages)
File "C: \Users\kojob\OneDrive\Desktop\AI_EOS\src\Llm.py", Line 18, in chat_complete
resp = client.chat.completions.create(
model=OPENAI_CHAT_MODEL, messages=messages,
