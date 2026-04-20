AI Healthcare Research Assistant
Built an intelligent healthcare research agent using AutoGen and Gemini that supports smart query handling, summarization, session memory, and comparison of results. Integrated a Streamlit-based UI for easy browser interaction and added human-in-the-loop (HITL) features for critical tasks like summary generation, subtopic creation, and report download.
=======
MediSyn Labs — AI Healthcare Research Assistant
Built with AutoGen + LangGraph + Gemini

---

## 📖 Project Overview

MediSyn Labs needed a way to stop medical researchers spending hours manually reading papers.
This AI assistant lets them type a clinical question and instantly get:
- Evidence-based literature summaries
- Side-by-side treatment comparisons
- Clinical Q&A with evidence levels
- PubMed article fetching (free API)
- Human-in-the-Loop approval before saving findings
- Downloadable PDF research reports

---

## 🗂️ Project Structure

```
medisyn_agent/
│
├── memory/
│   └── memory_schema.py        ← AgentState, MemoryManager, short/long-term memory
│
├── agent/
│   └── research_agent.py       ← AutoGen agents + LangGraph StateGraph
│
├── tools/
│   ├── medical_tools.py         ← 6 research tools (search, summarize, compare, PubMed)
│   └── report_generator.py     ← PDF/TXT report generation
│
├── ui/
│   └── app.py                  ← Streamlit 5-page web interface
│
├── knowledge/                  ← Long-term memory JSON stored here
├── outputs/                    ← Downloaded reports saved here
├── requirements.txt
├── .env.example → rename to .env
└── README.md
```

---

## ⚡ Setup (Windows / VS Code)

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure .env
copy .env.example .env
# Add: GOOGLE_API_KEY=your_key_here
# Get free key: https://aistudio.google.com/app/apikey

# 4. Run the app
streamlit run ui/app.py
```

---

## 🏗️ Architecture

### AutoGen Agents
```
MediSynAssistant (AssistantAgent)
  - Gemini 1.5 Flash LLM
  - System prompt: expert medical researcher
  - Tools: all 6 medical tools
  - max_consecutive_auto_reply: 3

ResearcherProxy (UserProxyAgent)
  - Represents the human researcher
  - human_input_mode: NEVER (automated)
  - Terminates on "TERMINATE"
```

### LangGraph StateGraph
```
[START]
   ↓
validate_node     → filters greetings, vague queries
   ↓ (if valid)
retrieve_node     → checks long-term memory for context
   ↓
research_node     → routes to correct tool + Gemini response
   ↓
subtopic_node     → generates 6 related research subtopics
   ↓
hitl_node         → sets awaiting_approval=True
   ↓
[UI shows approve/reject buttons]
   ↓ (if approved)
save_memory_node  → persists to long_term_memory.json
   ↓
[END]
```

### AgentState (Memory Schema)
```python
AgentState = {
  # Metadata
  researcher_id, project_id, disease_focus, session_id,

  # Short-term memory (max 7 entries — trimmed automatically)
  messages: List[ResearchMessage],

  # Long-term memory (JSON file — persists across sessions)
  long_term_memory: List[LongTermEntry],

  # Current interaction
  current_query, current_response, query_type, subtopics,

  # HITL control
  awaiting_approval, approved, hitl_feedback,

  # Report
  report_sections,

  # Validation
  is_valid_query, error_message,
}
```

---

## 🛠️ Tools

| Tool | Description | API |
|---|---|---|
| `search_medical_literature` | Web search for papers & studies | DuckDuckGo (free) |
| `summarize_literature` | Structured literature summary | Gemini |
| `compare_treatments` | Side-by-side drug/therapy comparison | Gemini |
| `fetch_pubmed_abstracts` | Real PubMed article titles & PMIDs | NCBI API (free) |
| `generate_subtopics` | Related research subtopics in JSON | Gemini |
| `clinical_qa` | Evidence-based clinical Q&A | Gemini |

---

## 📱 Streamlit UI Pages

| Page | Features |
|---|---|
| 🔬 Research | Query input, AI response, HITL approval, subtopics, add to report |
| ⚖️ Compare | Treatment A vs B comparison with presets |
| 💾 Memory Bank | Browse/search all approved long-term memories |
| 📋 Report Builder | Build + download PDF/TXT research report |
| ⚙️ Settings | Researcher profile, session stats |

---

## 🔑 Free APIs Used

- **Google Gemini 1.5 Flash** — Free tier (LLM)
- **DuckDuckGo Search** — Free, no key needed
- **NCBI PubMed E-utilities** — Free, no key needed
>>>>>>> 9bb6348 (First Version)
