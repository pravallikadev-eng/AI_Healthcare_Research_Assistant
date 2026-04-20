"""
tools/medical_tools.py
-----------------------
Tools available to the MediSyn AutoGen agents.

Tools:
  1. search_medical_literature  - DuckDuckGo search for medical papers
  2. summarize_literature       - Gemini-powered literature summarization
  3. compare_treatments         - Side-by-side therapy comparison
  4. fetch_pubmed_abstracts     - Fetches PubMed search results (free API)
  5. generate_subtopics         - Suggests related research subtopics
  6. retrieve_from_memory       - Searches long-term memory for past findings
  7. clinical_qa                - Answers specific clinical questions
"""

import os
import json
import re
import requests
from typing import List, Dict
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from duckduckgo_search import DDGS

load_dotenv()

MEDICAL_SYSTEM_PROMPT = """You are MediSyn, an expert AI medical research assistant at MediSyn Labs.
You assist clinical researchers, healthcare analysts, and medical professionals.
Always:
- Base responses on established medical knowledge and published research
- Clearly distinguish between established findings and emerging/uncertain evidence
- Include relevant caveats about individual patient variation
- Mention when consulting a healthcare professional is advisable
- Use precise medical terminology while remaining accessible
- Never diagnose or prescribe — your role is research assistance only
"""

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_tokens=2048,
    )


# ──────────────────────────────────────────────────
# Tool 1: Search Medical Literature (DuckDuckGo)
# ──────────────────────────────────────────────────

@tool("search_medical_literature")
def search_medical_literature_tool(query: str) -> str:
    """
    Searches the web for recent medical literature, clinical studies,
    and research papers related to the given medical topic.
    Uses DuckDuckGo — no API key required.

    Args:
        query: Medical topic or research question

    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    try:
        results = []
        search_query = f"{query} clinical study research pubmed 2023 2024"
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=6):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"Source: {r.get('href', 'N/A')}\n"
                    f"Abstract: {r.get('body', 'N/A')}\n"
                )
        if not results:
            return f"No results found for: {query}"
        return "=== SEARCH RESULTS ===\n\n" + "\n---\n".join(results)
    except Exception as e:
        return f"Search error: {str(e)}. Using knowledge base only."


# ──────────────────────────────────────────────────
# Tool 2: Summarize Literature
# ──────────────────────────────────────────────────

@tool("summarize_literature")
def summarize_literature_tool(topic: str) -> str:
    """
    Generates a comprehensive, structured literature summary on a medical topic.
    Covers: overview, key findings, current evidence, limitations, and implications.

    Args:
        topic: Medical topic to summarize (e.g., "mRNA vaccine efficacy")

    Returns:
        Structured literature summary with sections
    """
    llm = get_llm()
    prompt = f"""Provide a comprehensive medical literature summary on: "{topic}"

Structure your response with these sections:

## Overview
[2-3 sentence overview of the topic]

## Key Findings from Recent Literature
[5-7 key evidence-based findings with brief explanations]

## Current Evidence Level
[Describe the strength and quality of available evidence]

## Clinical Implications
[How findings apply to clinical practice]

## Limitations & Research Gaps
[What is still unknown or debated]

## Key References (Illustrative)
[Mention 3-4 types of studies that would be relevant, e.g., "RCTs comparing X with Y"]

Always base responses on established medical knowledge. Flag any emerging/uncertain areas."""

    try:
        response = llm.invoke([
            SystemMessage(content=MEDICAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content
    except Exception as e:
        return f"Summarization error: {str(e)}"


# ──────────────────────────────────────────────────
# Tool 3: Compare Treatments
# ──────────────────────────────────────────────────

@tool("compare_treatments")
def compare_treatments_tool(treatment_a: str, treatment_b: str, condition: str = "") -> str:
    """
    Generates a detailed side-by-side comparison of two medical treatments,
    drugs, or therapeutic approaches for a given condition.

    Args:
        treatment_a: First treatment/drug name
        treatment_b: Second treatment/drug name
        condition  : Medical condition being treated (optional but recommended)

    Returns:
        Structured comparison table and analysis
    """
    llm = get_llm()
    context = f"for {condition}" if condition else ""
    prompt = f"""Compare {treatment_a} vs {treatment_b} {context} for medical researchers.

Provide a comprehensive comparison covering:

## Mechanism of Action
- {treatment_a}: [mechanism]
- {treatment_b}: [mechanism]

## Efficacy
- {treatment_a}: [efficacy data, key trial results]
- {treatment_b}: [efficacy data, key trial results]

## Safety Profile & Side Effects
- {treatment_a}: [common/serious adverse effects]
- {treatment_b}: [common/serious adverse effects]

## Dosing & Administration
- {treatment_a}: [typical dosing]
- {treatment_b}: [typical dosing]

## Patient Population Considerations
[Who benefits most from each? Contraindications?]

## Cost & Accessibility
[General comparison of availability/cost]

## Current Clinical Guidelines
[What major guidelines (e.g., WHO, FDA, EMA) recommend]

## Summary Verdict
[Evidence-based summary of when each is preferred]

Note: This is for research purposes only. Clinical decisions require physician judgment."""

    try:
        response = llm.invoke([
            SystemMessage(content=MEDICAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content
    except Exception as e:
        return f"Comparison error: {str(e)}"


# ──────────────────────────────────────────────────
# Tool 4: PubMed Search (Free API, no key needed)
# ──────────────────────────────────────────────────

@tool("fetch_pubmed_abstracts")
def fetch_pubmed_abstracts_tool(query: str, max_results: int = 5) -> str:
    """
    Fetches real PubMed article abstracts using the free NCBI E-utilities API.
    No API key required for basic usage.

    Args:
        query      : Medical search query
        max_results: Number of abstracts to fetch (max 10)

    Returns:
        Formatted list of PubMed abstracts with titles and PMIDs
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    try:
        # Step 1: Search for article IDs
        search_url = f"{base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 10),
            "retmode": "json",
            "sort": "relevance",
        }
        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_resp.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return f"No PubMed articles found for: '{query}'"

        # Step 2: Fetch article summaries
        summary_url = f"{base_url}/esummary.fcgi"
        summary_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        summary_resp = requests.get(summary_url, params=summary_params, timeout=10)
        summary_data = summary_resp.json()

        results = []
        for pmid in pmids:
            article = summary_data.get("result", {}).get(pmid, {})
            title = article.get("title", "No title")
            pub_date = article.get("pubdate", "Unknown date")
            authors = article.get("authors", [])
            author_names = ", ".join([a.get("name", "") for a in authors[:3]])
            if len(authors) > 3:
                author_names += " et al."

            results.append(
                f"PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Authors: {author_names}\n"
                f"Published: {pub_date}\n"
                f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            )

        return f"=== PUBMED RESULTS for '{query}' ===\n\n" + "\n---\n".join(results)

    except requests.exceptions.ConnectionError:
        return f"PubMed API unavailable (no internet). Using knowledge base for: {query}"
    except Exception as e:
        return f"PubMed fetch error: {str(e)}"


# ──────────────────────────────────────────────────
# Tool 5: Generate Subtopics
# ──────────────────────────────────────────────────

@tool("generate_subtopics")
def generate_subtopics_tool(topic: str) -> str:
    """
    Generates a list of relevant research subtopics for a given medical topic.
    Useful for researchers who want to explore related areas.

    Args:
        topic: Main medical topic or research question

    Returns:
        JSON string with categorized subtopics
    """
    llm = get_llm()
    prompt = f"""For the medical research topic: "{topic}"

Generate a structured list of research subtopics in JSON format:
{{
  "primary_subtopics": ["subtopic 1", "subtopic 2", "subtopic 3", "subtopic 4", "subtopic 5"],
  "clinical_angles": ["clinical angle 1", "clinical angle 2", "clinical angle 3"],
  "research_gaps": ["gap 1", "gap 2", "gap 3"],
  "related_conditions": ["condition 1", "condition 2", "condition 3"],
  "suggested_queries": [
    "Suggested research query 1?",
    "Suggested research query 2?",
    "Suggested research query 3?"
  ]
}}

Return ONLY valid JSON, no other text."""

    try:
        response = llm.invoke([
            SystemMessage(content=MEDICAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        return raw
    except Exception as e:
        return json.dumps({
            "primary_subtopics": [f"{topic} mechanisms", f"{topic} treatments", f"{topic} diagnosis"],
            "clinical_angles": ["Treatment outcomes", "Patient safety", "Drug interactions"],
            "research_gaps": ["Long-term data needed", "Pediatric studies limited"],
            "related_conditions": [],
            "suggested_queries": [f"What are the latest treatments for {topic}?"],
        })


# ──────────────────────────────────────────────────
# Tool 6: Clinical Q&A
# ──────────────────────────────────────────────────

@tool("clinical_qa")
def clinical_qa_tool(question: str) -> str:
    """
    Answers specific clinical or pharmacological questions.
    Covers drug mechanisms, dosing, interactions, contraindications,
    and evidence-based treatment protocols.

    Args:
        question: Specific clinical or pharmacological question

    Returns:
        Evidence-based clinical answer with confidence indicators
    """
    llm = get_llm()
    prompt = f"""Answer this clinical/research question: "{question}"

Provide a structured answer:

## Answer
[Direct, evidence-based answer]

## Evidence Base
[Quality and level of supporting evidence]

## Important Caveats
[Key limitations, individual variation, or uncertain areas]

## Clinical Context
[When this applies / when it doesn't]

Be precise, cite evidence levels (e.g., Level 1 RCT, meta-analysis, observational data).
Always note if the evidence is preliminary or emerging."""

    try:
        response = llm.invoke([
            SystemMessage(content=MEDICAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content
    except Exception as e:
        return f"Clinical Q&A error: {str(e)}"


# ──────────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────────

MEDICAL_TOOLS = [
    search_medical_literature_tool,
    summarize_literature_tool,
    compare_treatments_tool,
    fetch_pubmed_abstracts_tool,
    generate_subtopics_tool,
    clinical_qa_tool,
]
