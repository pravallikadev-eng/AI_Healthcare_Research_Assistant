"""
memory/memory_schema.py
------------------------
Defines the AgentState (Healthcare Memory Schema) for MediSyn Labs.

Short-term memory  : Active session queries and responses (max 7)
Long-term memory   : Stored summaries, comparisons, literature findings (persistent JSON)
Metadata           : researcher_id, project_id, disease_focus, session_id

Also implements:
  - MemoryManager  : Handles read/write for both memory types
  - State reducers : Update memory with each new interaction
  - Message filter : Removes greetings/vague queries before LLM call
"""

import os
import json
import uuid
from typing import TypedDict, Annotated, List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────
# 1. AgentState — Healthcare Memory Schema
# ──────────────────────────────────────────────────

class ResearchMessage(TypedDict):
    """A single query-response pair in memory."""
    role: str            # "user" or "assistant"
    content: str         # The message text
    timestamp: str       # ISO format
    query_type: str      # "summarization" | "comparison" | "qa" | "unknown"
    approved: bool       # HITL approval status


class LongTermEntry(TypedDict):
    """A single long-term memory record (persisted across sessions)."""
    entry_id: str
    researcher_id: str
    project_id: str
    disease_focus: str
    query: str
    summary: str
    query_type: str
    timestamp: str
    approved: bool
    tags: List[str]


class AgentState(TypedDict):
    """
    Master state object flowing through the StateGraph.
    
    Short-term : messages (list of last N queries — trimmed to SHORT_TERM_LIMIT)
    Long-term  : long_term_memory (persisted to JSON file)
    Metadata   : researcher_id, project_id, disease_focus
    Control    : current_query, current_response, awaiting_approval, approved
    """
    # ── Metadata ──────────────────────────────────
    researcher_id: str
    project_id: str
    disease_focus: str
    session_id: str

    # ── Short-term memory (active session) ────────
    messages: List[ResearchMessage]         # trimmed to SHORT_TERM_LIMIT

    # ── Long-term memory (persistent) ─────────────
    long_term_memory: List[LongTermEntry]   # loaded from / saved to JSON

    # ── Current interaction ────────────────────────
    current_query: str                      # latest user input
    current_response: str                   # latest LLM response
    query_type: str                         # detected query type
    subtopics: List[str]                    # suggested subtopics

    # ── HITL (Human-in-the-Loop) ──────────────────
    awaiting_approval: bool                 # True if waiting for human review
    approved: Optional[bool]               # True=approved, False=rejected, None=pending
    hitl_feedback: str                      # human feedback/edits

    # ── Report ────────────────────────────────────
    report_content: str                     # accumulated report text
    report_sections: List[Dict]             # structured sections for download

    # ── Control flow ──────────────────────────────
    is_valid_query: bool                    # False if greeting/vague
    error_message: str                      # any error to display


# ──────────────────────────────────────────────────
# 2. State Reducer Functions
# ──────────────────────────────────────────────────

SHORT_TERM_LIMIT = int(os.getenv("SHORT_TERM_MEMORY_LIMIT", 7))

# Non-informational patterns to filter out
VAGUE_PATTERNS = [
    "hello", "hi", "hey", "good morning", "good evening",
    "thanks", "thank you", "bye", "goodbye", "ok", "okay",
    "sure", "yes", "no", "help", "test", "testing",
]

def is_valid_medical_query(query: str) -> bool:
    """
    Returns True if the query is a substantive medical/research question.
    Filters out greetings, vague inputs, and one-word queries.
    """
    query_lower = query.lower().strip()
    if len(query_lower) < 15:
        return False
    if any(query_lower.startswith(pat) for pat in VAGUE_PATTERNS):
        return False
    if query_lower in VAGUE_PATTERNS:
        return False
    return True


def detect_query_type(query: str) -> str:
    """
    Detects the type of medical query for routing and labeling.
    
    Returns: "comparison" | "summarization" | "clinical_qa" | "trial_search" | "general"
    """
    q = query.lower()
    if any(w in q for w in ["compare", "versus", "vs", "difference between", "better than", "vs."]):
        return "comparison"
    elif any(w in q for w in ["summarize", "summary", "overview", "what is", "explain", "describe"]):
        return "summarization"
    elif any(w in q for w in ["trial", "clinical trial", "study", "research", "published", "literature"]):
        return "trial_search"
    elif any(w in q for w in ["treatment", "drug", "therapy", "medication", "dose", "dosage", "side effect"]):
        return "clinical_qa"
    return "general"


def trim_messages(messages: List[ResearchMessage]) -> List[ResearchMessage]:
    """Trims short-term memory to the last SHORT_TERM_LIMIT entries."""
    return messages[-SHORT_TERM_LIMIT:] if len(messages) > SHORT_TERM_LIMIT else messages


def add_message_to_state(
    state: AgentState,
    role: str,
    content: str,
    query_type: str = "general",
    approved: bool = True,
) -> List[ResearchMessage]:
    """Adds a new message and returns trimmed list."""
    new_msg: ResearchMessage = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "query_type": query_type,
        "approved": approved,
    }
    updated = list(state.get("messages", [])) + [new_msg]
    return trim_messages(updated)


# ──────────────────────────────────────────────────
# 3. MemoryManager — Handles Long-Term Persistence
# ──────────────────────────────────────────────────

LONG_TERM_PATH = os.getenv("LONG_TERM_MEMORY_PATH", "./memory/long_term_memory.json")


class MemoryManager:
    """
    Manages both short-term (in-memory) and long-term (JSON file) memory.
    
    Long-term memory persists across sessions — researchers can retrieve
    past summaries even after restarting the application.
    """

    def __init__(self, filepath: str = LONG_TERM_PATH):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._data: List[LongTermEntry] = self._load()

    def _load(self) -> List[LongTermEntry]:
        """Loads long-term memory from JSON file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        """Persists long-term memory to JSON file."""
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    def save_entry(
        self,
        researcher_id: str,
        project_id: str,
        disease_focus: str,
        query: str,
        summary: str,
        query_type: str = "general",
        approved: bool = True,
        tags: List[str] = None,
    ) -> LongTermEntry:
        """Saves a new research finding to long-term memory."""
        entry: LongTermEntry = {
            "entry_id": str(uuid.uuid4())[:8],
            "researcher_id": researcher_id,
            "project_id": project_id,
            "disease_focus": disease_focus,
            "query": query,
            "summary": summary,
            "query_type": query_type,
            "timestamp": datetime.now().isoformat(),
            "approved": approved,
            "tags": tags or [],
        }
        self._data.append(entry)
        self._save()
        return entry

    def get_all(self) -> List[LongTermEntry]:
        """Returns all long-term memory entries."""
        return self._data

    def get_by_researcher(self, researcher_id: str) -> List[LongTermEntry]:
        return [e for e in self._data if e["researcher_id"] == researcher_id]

    def get_by_disease(self, disease_focus: str) -> List[LongTermEntry]:
        keyword = disease_focus.lower()
        return [e for e in self._data if keyword in e["disease_focus"].lower() or keyword in e["query"].lower()]

    def search(self, keyword: str) -> List[LongTermEntry]:
        """Searches long-term memory by keyword in query or summary."""
        kw = keyword.lower()
        return [
            e for e in self._data
            if kw in e["query"].lower() or kw in e["summary"].lower()
               or any(kw in t.lower() for t in e.get("tags", []))
        ]

    def delete_entry(self, entry_id: str) -> bool:
        before = len(self._data)
        self._data = [e for e in self._data if e["entry_id"] != entry_id]
        if len(self._data) < before:
            self._save()
            return True
        return False

    def count(self) -> int:
        return len(self._data)

    def get_recent(self, n: int = 5) -> List[LongTermEntry]:
        return sorted(self._data, key=lambda x: x["timestamp"], reverse=True)[:n]


# Singleton instance
memory_manager = MemoryManager()
