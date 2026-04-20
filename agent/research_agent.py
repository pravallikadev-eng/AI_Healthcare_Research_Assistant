"""
agent/research_agent.py
------------------------
Core AI Healthcare Research Assistant using AutoGen + LangGraph.

Architecture:
  AutoGen AssistantAgent  ←→  UserProxyAgent
          │
          ▼
  LangGraph StateGraph
    ├── validate_node      : Filter vague/greeting queries
    ├── retrieve_node      : Check long-term memory for past findings
    ├── research_node      : Route to correct tool (summarize/compare/qa)
    ├── subtopic_node      : Generate research subtopics
    ├── hitl_node          : Flag for Human-in-the-Loop approval
    └── save_memory_node   : Persist approved findings to long-term memory

AutoGen Agents:
  MediSynAssistant  : Gemini-powered medical research agent
  ResearchProxy     : Executes tools, manages conversation flow
"""

import os
import sys
import json
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.memory_schema import (
    AgentState, MemoryManager, memory_manager,
    is_valid_medical_query, detect_query_type,
    add_message_to_state, trim_messages,
)
from tools.medical_tools import (
    search_medical_literature_tool,
    summarize_literature_tool,
    compare_treatments_tool,
    fetch_pubmed_abstracts_tool,
    generate_subtopics_tool,
    clinical_qa_tool,
)

# ── AutoGen Setup ──────────────────────────────────────────────────────────────
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("⚠️  AutoGen not installed. Using LangGraph-only mode.")


MEDICAL_SYSTEM_PROMPT = """You are MediSyn, an expert AI Healthcare Research Assistant at MediSyn Labs.
You assist medical researchers, clinicians, and healthcare analysts with:
- Summarizing medical literature and clinical studies
- Comparing treatment options and therapies
- Answering evidence-based clinical questions
- Identifying research gaps and subtopics

Always base responses on established medical knowledge.
Clearly flag uncertainty or emerging evidence.
Never diagnose, prescribe, or replace clinical judgment."""


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_tokens=2048,
    )


# ── AutoGen Agent Factory ──────────────────────────────────────────────────────

def build_autogen_agents():
    """
    Builds AutoGen AssistantAgent and UserProxyAgent configured with Gemini.
    
    AutoGen Pattern:
      UserProxyAgent ←message→ AssistantAgent (MediSyn)
                         ↓
                  Executes tools / generates responses
    """
    if not AUTOGEN_AVAILABLE:
        return None, None

    api_key = os.getenv("GOOGLE_API_KEY", "")
    config_list = [{
        "model": "gemini-1.5-flash",
        "api_key": api_key,
        "api_type": "google",
    }]

    llm_config = {
        "config_list": config_list,
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    # MediSyn Research Assistant Agent
    assistant = AssistantAgent(
        name="MediSynAssistant",
        system_message=MEDICAL_SYSTEM_PROMPT + "\n\nProvide thorough, evidence-based responses. Always structure your answers with clear sections.",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
    )

    # User Proxy Agent (represents the researcher)
    user_proxy = UserProxyAgent(
        name="ResearcherProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        default_auto_reply="Please continue with the research analysis.",
        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
    )

    return assistant, user_proxy


# ── LangGraph Node Functions ───────────────────────────────────────────────────

def validate_node(state: AgentState) -> AgentState:
    """
    Node 1: Validates if the query is a substantive medical question.
    Filters greetings, vague queries, and very short inputs.
    """
    query = state.get("current_query", "").strip()

    if not query:
        return {**state, "is_valid_query": False, "error_message": "Empty query received."}

    if is_valid_medical_query(query):
        query_type = detect_query_type(query)
        return {
            **state,
            "is_valid_query": True,
            "query_type": query_type,
            "error_message": "",
        }
    else:
        return {
            **state,
            "is_valid_query": False,
            "error_message": (
                "Your query appears to be a greeting or too vague for medical research. "
                "Please ask a specific medical or clinical question, e.g., "
                "'Summarize recent literature on metformin for type 2 diabetes.'"
            ),
        }


def retrieve_node(state: AgentState) -> AgentState:
    """
    Node 2: Checks long-term memory for relevant past findings.
    Prepends context to the current query if found.
    """
    query = state.get("current_query", "")
    results = memory_manager.search(query)

    if results:
        memory_context = f"\n\n[Past Research Context — {len(results)} relevant findings found]\n"
        for r in results[:2]:  # Top 2 relevant past findings
            memory_context += f"- [{r['query_type']}] {r['query'][:80]}... (saved {r['timestamp'][:10]})\n"
        return {**state, "current_query": query + memory_context}

    return state


def research_node(state: AgentState) -> AgentState:
    """
    Node 3: Core research node. Routes to the correct tool based on query_type,
    then generates a comprehensive response using Gemini.
    """
    llm = get_llm()
    query = state.get("current_query", "")
    query_type = state.get("query_type", "general")

    tool_result = ""
    response = ""

    try:
        # Route to appropriate tool
        if query_type == "comparison":
            # Extract two treatments from query
            parts = query.split(" vs " if " vs " in query.lower() else " versus ")
            if len(parts) >= 2:
                t_a = parts[0].strip().split()[-1]  # Last word before 'vs'
                t_b = parts[1].strip().split()[0]   # First word after 'vs'
                tool_result = compare_treatments_tool.invoke(f"{t_a} {t_b}")
            else:
                tool_result = search_medical_literature_tool.invoke(query)

        elif query_type == "trial_search":
            pubmed_result = fetch_pubmed_abstracts_tool.invoke(query)
            search_result = search_medical_literature_tool.invoke(query)
            tool_result = pubmed_result + "\n\n" + search_result

        elif query_type == "summarization":
            tool_result = summarize_literature_tool.invoke(query)

        elif query_type == "clinical_qa":
            tool_result = clinical_qa_tool.invoke(query)

        else:
            # General: search + summarize
            search_result = search_medical_literature_tool.invoke(query)
            tool_result = search_result

        # Generate final response using Gemini with tool context
        final_prompt = f"""Research Query: {query}

Research Data Gathered:
{tool_result[:3000]}

Based on the above research data, provide a comprehensive, well-structured response for a medical researcher.
Include key findings, evidence quality, clinical implications, and any important caveats.
Format your response with clear markdown headings."""

        llm_response = llm.invoke([
            SystemMessage(content=MEDICAL_SYSTEM_PROMPT),
            HumanMessage(content=final_prompt),
        ])
        response = llm_response.content

    except Exception as e:
        response = f"Research error: {str(e)}\n\nFallback: Please try a more specific query."

    # Update messages
    updated_messages = add_message_to_state(state, "user", state.get("current_query", ""), query_type)
    updated_messages = trim_messages(updated_messages + [{
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
        "query_type": query_type,
        "approved": False,  # Will be set by HITL
    }])

    return {
        **state,
        "current_response": response,
        "messages": updated_messages,
        "awaiting_approval": True,
    }


def subtopic_node(state: AgentState) -> AgentState:
    """
    Node 4: Generates research subtopics for the current query.
    Runs in parallel with research_node context.
    """
    query = state.get("current_query", "").split("\n")[0]  # Original query only
    try:
        subtopics_json = generate_subtopics_tool.invoke(query)
        subtopics_data = json.loads(subtopics_json)
        all_subtopics = (
            subtopics_data.get("primary_subtopics", []) +
            subtopics_data.get("suggested_queries", [])
        )
        return {**state, "subtopics": all_subtopics[:6]}
    except Exception:
        return {**state, "subtopics": []}


def hitl_node(state: AgentState) -> AgentState:
    """
    Node 5: Human-in-the-Loop checkpoint.
    In Streamlit UI, this is handled interactively.
    In CLI mode, auto-approves (can be overridden).
    Sets awaiting_approval = True — UI must set approved = True/False.
    """
    # This node marks the response as needing HITL approval
    # The Streamlit UI reads awaiting_approval and shows approval buttons
    return {**state, "awaiting_approval": True, "approved": None}


def save_memory_node(state: AgentState) -> AgentState:
    """
    Node 6: Saves approved findings to long-term memory.
    Only runs if approved = True.
    """
    if not state.get("approved"):
        return state

    try:
        entry = memory_manager.save_entry(
            researcher_id=state.get("researcher_id", "default"),
            project_id=state.get("project_id", "general"),
            disease_focus=state.get("disease_focus", "general"),
            query=state.get("current_query", "").split("\n")[0][:200],
            summary=state.get("current_response", "")[:1000],
            query_type=state.get("query_type", "general"),
            approved=True,
            tags=[state.get("disease_focus", ""), state.get("query_type", "")],
        )
        print(f"✅ Saved to long-term memory: entry_id={entry['entry_id']}")
    except Exception as e:
        print(f"⚠️  Memory save error: {e}")

    return state


# ── Routing Logic ─────────────────────────────────────────────────────────────

def route_after_validate(state: AgentState) -> Literal["retrieve_node", "end_invalid"]:
    if state.get("is_valid_query"):
        return "retrieve_node"
    return "end_invalid"


def route_after_hitl(state: AgentState) -> Literal["save_memory_node", "end_approved"]:
    """After HITL, if approved save to memory. Otherwise just end."""
    if state.get("approved") is True:
        return "save_memory_node"
    return "end_approved"


# ── Build StateGraph ──────────────────────────────────────────────────────────

def build_research_graph():
    """
    Builds and compiles the full LangGraph StateGraph for MediSyn.
    
    Flow:
      START → validate → (if valid) → retrieve → research → subtopic
                                                      ↓
                                                   hitl_node
                                                      ↓
                                          approved? → save_memory → END
                                          rejected?            → END
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("validate_node", validate_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("research_node", research_node)
    graph.add_node("subtopic_node", subtopic_node)
    graph.add_node("hitl_node", hitl_node)
    graph.add_node("save_memory_node", save_memory_node)

    # Entry point
    graph.add_edge(START, "validate_node")

    # Conditional routing after validation
    graph.add_conditional_edges(
        "validate_node",
        route_after_validate,
        {
            "retrieve_node": "retrieve_node",
            "end_invalid": END,
        }
    )

    # Sequential flow
    graph.add_edge("retrieve_node", "research_node")
    graph.add_edge("research_node", "subtopic_node")
    graph.add_edge("subtopic_node", "hitl_node")

    # HITL conditional
    graph.add_conditional_edges(
        "hitl_node",
        route_after_hitl,
        {
            "save_memory_node": "save_memory_node",
            "end_approved": END,
        }
    )
    graph.add_edge("save_memory_node", END)

    return graph.compile()


# ── MediSynResearchAgent Class ────────────────────────────────────────────────

class MediSynResearchAgent:
    """
    Main entry point for the MediSyn Healthcare Research Assistant.
    Combines AutoGen agents with LangGraph StateGraph.
    """

    def __init__(
        self,
        researcher_id: str = None,
        project_id: str = None,
        disease_focus: str = None,
    ):
        self.researcher_id = researcher_id or os.getenv("DEFAULT_RESEARCHER_ID", "researcher_001")
        self.project_id = project_id or os.getenv("DEFAULT_PROJECT_ID", "project_general")
        self.disease_focus = disease_focus or os.getenv("DEFAULT_DISEASE_FOCUS", "general")

        self.graph = build_research_graph()
        self.memory = memory_manager

        # Build AutoGen agents
        self.assistant, self.user_proxy = build_autogen_agents()

        print(f"\n✅ MediSyn Research Agent initialized")
        print(f"   Researcher: {self.researcher_id}")
        print(f"   Project: {self.project_id}")
        print(f"   Disease Focus: {self.disease_focus}")
        print(f"   Long-term memory: {self.memory.count()} entries")

    def create_initial_state(self) -> AgentState:
        """Creates a fresh AgentState with researcher metadata."""
        return AgentState(
            researcher_id=self.researcher_id,
            project_id=self.project_id,
            disease_focus=self.disease_focus,
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            messages=[],
            long_term_memory=self.memory.get_all(),
            current_query="",
            current_response="",
            query_type="general",
            subtopics=[],
            awaiting_approval=False,
            approved=None,
            hitl_feedback="",
            report_content="",
            report_sections=[],
            is_valid_query=True,
            error_message="",
        )

    def run(self, query: str, auto_approve: bool = False) -> AgentState:
        """
        Runs a single research query through the full pipeline.
        
        Args:
            query       : The research question
            auto_approve: If True, skips HITL (for CLI/testing use)
        
        Returns:
            Final AgentState with response, subtopics, and memory updates
        """
        state = self.create_initial_state()
        state["current_query"] = query

        # Run the graph
        final_state = self.graph.invoke(state)

        # Handle HITL
        if final_state.get("awaiting_approval") and auto_approve:
            final_state["approved"] = True
            final_state = self.graph.invoke({**final_state, "awaiting_approval": False})

        return final_state

    def approve_and_save(self, state: AgentState, feedback: str = "") -> AgentState:
        """Called by UI when researcher approves a response."""
        updated = {
            **state,
            "approved": True,
            "awaiting_approval": False,
            "hitl_feedback": feedback,
        }
        # Save to memory
        self.memory.save_entry(
            researcher_id=state.get("researcher_id", "default"),
            project_id=state.get("project_id", "general"),
            disease_focus=state.get("disease_focus", "general"),
            query=state.get("current_query", "").split("\n")[0][:200],
            summary=state.get("current_response", "")[:1000],
            query_type=state.get("query_type", "general"),
            approved=True,
        )
        return updated

    def reject_response(self, state: AgentState) -> AgentState:
        """Called by UI when researcher rejects a response."""
        return {**state, "approved": False, "awaiting_approval": False}
