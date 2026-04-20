"""
ui/app.py
----------
Streamlit UI for the MediSyn Labs AI Healthcare Research Assistant.

Pages:
  1. 🔬 Research     - Main query interface with HITL approval
  2. ⚖️  Compare      - Side-by-side treatment comparison
  3. 💾 Memory       - Browse long-term research memory
  4. 📋 Report       - Build and download research reports
  5. ⚙️  Settings     - Researcher profile configuration

Run: streamlit run ui/app.py
"""

import os
import sys
import json
from datetime import datetime
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediSyn Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    .main-header {
        background: linear-gradient(90deg, #0a3d62, #1e6fa5);
        color: white; padding: 20px 28px; border-radius: 12px;
        margin-bottom: 20px;
    }
    .response-box {
        background: white; border-left: 4px solid #1e6fa5;
        border-radius: 8px; padding: 20px; margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .hitl-box {
        background: #fffbf0; border: 2px solid #f0a500;
        border-radius: 10px; padding: 16px; margin: 12px 0;
    }
    .memory-card {
        background: white; border-radius: 8px; padding: 14px;
        margin: 8px 0; border: 1px solid #dde4ed;
    }
    .approved-badge { color: #27ae60; font-weight: bold; }
    .pending-badge  { color: #e67e22; font-weight: bold; }
    .subtopic-chip {
        display:inline-block; background:#e8f0fe; color:#1a73e8;
        border-radius:16px; padding:4px 12px; margin:3px;
        font-size:0.82em; cursor:pointer;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
for key, default in {
    "agent": None,
    "agent_ready": False,
    "current_state": None,
    "chat_history": [],
    "report_sections": [],
    "researcher_id": "researcher_001",
    "project_id": "project_general",
    "disease_focus": "general",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource(show_spinner="Initializing MediSyn Agent...")
def load_agent(researcher_id, project_id, disease_focus):
    from agent.research_agent import MediSynResearchAgent
    return MediSynResearchAgent(
        researcher_id=researcher_id,
        project_id=project_id,
        disease_focus=disease_focus,
    )


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 MediSyn Labs")
    st.markdown("*AI Healthcare Research Assistant*")
    st.markdown("*Powered by AutoGen + Gemini*")
    st.divider()

    # API Key
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_google_api_key_here":
        st.error("⚠️ GOOGLE_API_KEY not set")
        key_input = st.text_input("Enter Gemini API Key:", type="password")
        if key_input:
            os.environ["GOOGLE_API_KEY"] = key_input
            st.success("✅ Key set!")
    else:
        st.success("✅ Gemini API Key loaded")

    st.divider()

    # Researcher Profile
    st.markdown("### 👤 Researcher Profile")
    st.session_state.researcher_id = st.text_input("Researcher ID:", value=st.session_state.researcher_id)
    st.session_state.project_id = st.text_input("Project ID:", value=st.session_state.project_id)
    st.session_state.disease_focus = st.text_input("Disease Focus:", value=st.session_state.disease_focus, placeholder="e.g., diabetes, oncology")

    st.divider()
    page = st.radio("Navigate", ["🔬 Research", "⚖️ Compare Treatments", "💾 Memory Bank", "📋 Report Builder", "⚙️ Settings"])

    st.divider()
    if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
        with st.spinner("Loading MediSyn Agent..."):
            try:
                agent = load_agent(
                    st.session_state.researcher_id,
                    st.session_state.project_id,
                    st.session_state.disease_focus,
                )
                st.session_state.agent = agent
                st.session_state.agent_ready = True
                st.success(f"✅ Agent Ready! ({agent.memory.count()} memories)")
            except Exception as e:
                st.error(f"❌ {e}")

    status = "🟢 Online" if st.session_state.agent_ready else "🔴 Offline"
    st.markdown(f"**Agent Status:** {status}")


# ─────────────────────────────────────────────
# PAGE 1: Research
# ─────────────────────────────────────────────
if page == "🔬 Research":
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0">🔬 MediSyn Research Assistant</h2>
        <p style="margin:4px 0 0">AI-powered medical literature analysis, summarization & clinical Q&A</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.agent_ready:
        st.warning("⚠️ Please initialize the agent using the sidebar button.")
        st.stop()

    # Sample queries
    st.markdown("**💡 Sample Queries:**")
    samples = [
        "Summarize recent literature on mRNA vaccines efficacy",
        "What are the latest treatments for type 2 diabetes?",
        "Clinical evidence for metformin vs SGLT2 inhibitors",
        "What does research say about immunotherapy for lung cancer?",
        "Explain the mechanism of CRISPR-based gene therapy",
    ]
    cols = st.columns(len(samples))
    for i, s in enumerate(samples):
        if cols[i].button(s[:35] + "...", use_container_width=True):
            st.session_state["prefill"] = s

    st.divider()

    # Query input
    query = st.text_area(
        "Enter your research query:",
        value=st.session_state.pop("prefill", ""),
        height=100,
        placeholder="e.g., Summarize recent literature on CRISPR gene editing for sickle cell disease",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        search_btn = st.button("🔍 Research", type="primary", use_container_width=True)
    with col2:
        auto_approve = st.checkbox("Auto-approve", value=False)

    if search_btn and query.strip():
        with st.spinner("🧠 MediSyn is researching..."):
            state = st.session_state.agent.run(query, auto_approve=auto_approve)
            st.session_state.current_state = state

    # Display Results
    state = st.session_state.current_state
    if state:
        if not state.get("is_valid_query"):
            st.warning(f"⚠️ {state.get('error_message', 'Invalid query')}")
        else:
            # Query type badge
            qtype = state.get("query_type", "general")
            type_colors = {
                "comparison": "🟠", "summarization": "🔵",
                "clinical_qa": "🟢", "trial_search": "🟣", "general": "⚪"
            }
            st.markdown(f"**Query Type:** {type_colors.get(qtype, '⚪')} `{qtype.replace('_', ' ').title()}`")

            # Response
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.markdown(state.get("current_response", "No response generated."))
            st.markdown('</div>', unsafe_allow_html=True)

            # Subtopics
            subtopics = state.get("subtopics", [])
            if subtopics:
                st.markdown("**🗂️ Related Research Subtopics:**")
                chips = " ".join([f'<span class="subtopic-chip">{s}</span>' for s in subtopics[:6]])
                st.markdown(chips, unsafe_allow_html=True)

            # HITL Approval Panel
            if not auto_approve and state.get("awaiting_approval"):
                st.markdown('<div class="hitl-box">', unsafe_allow_html=True)
                st.markdown("### 🔔 Human Review Required")
                st.markdown("*Please review this AI-generated medical summary before saving to your research memory.*")
                feedback = st.text_area("Optional feedback or edits:", key="hitl_feedback", height=80)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("✅ Approve & Save to Memory", type="primary", use_container_width=True):
                        updated = st.session_state.agent.approve_and_save(state, feedback)
                        st.session_state.current_state = updated
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": state.get("current_response", ""),
                            "approved": True,
                            "timestamp": datetime.now().isoformat(),
                        })
                        st.success("✅ Approved and saved to long-term memory!")
                        st.rerun()
                with col_b:
                    if st.button("✏️ Edit Response", use_container_width=True):
                        st.info("Edit the response above and click Approve.")
                with col_c:
                    if st.button("❌ Reject", use_container_width=True):
                        updated = st.session_state.agent.reject_response(state)
                        st.session_state.current_state = updated
                        st.warning("Response rejected. Not saved to memory.")
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            elif state.get("approved"):
                st.success("✅ This finding is approved and saved to research memory.")

            # Add to report button
            if st.button("📋 Add to Report"):
                st.session_state.report_sections.append({
                    "heading": query[:60] + "..." if len(query) > 60 else query,
                    "content": state.get("current_response", ""),
                })
                st.success(f"Added to report! ({len(st.session_state.report_sections)} sections)")

    # Session history
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### 📜 Session History")
        for i, h in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {h['query'][:70]}... {'✅' if h.get('approved') else ''}"):
                st.markdown(h["response"][:500] + "...")


# ─────────────────────────────────────────────
# PAGE 2: Compare Treatments
# ─────────────────────────────────────────────
elif page == "⚖️ Compare Treatments":
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0">⚖️ Treatment Comparison Engine</h2>
        <p style="margin:4px 0 0">Evidence-based side-by-side analysis of therapies, drugs, and interventions</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.agent_ready:
        st.warning("⚠️ Please initialize the agent first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        treatment_a = st.text_input("Treatment A:", placeholder="e.g., Remdesivir")
    with col2:
        treatment_b = st.text_input("Treatment B:", placeholder="e.g., Paxlovid")
    with col3:
        condition = st.text_input("Condition (optional):", placeholder="e.g., COVID-19")

    # Quick comparison presets
    st.markdown("**Quick Comparisons:**")
    presets = [
        ("Metformin", "SGLT2 inhibitors", "Type 2 diabetes"),
        ("Remdesivir", "Paxlovid", "COVID-19"),
        ("mRNA vaccine", "Traditional vaccine", "COVID-19"),
        ("Chemotherapy", "Immunotherapy", "Lung cancer"),
    ]
    preset_cols = st.columns(len(presets))
    for i, (a, b, c) in enumerate(presets):
        if preset_cols[i].button(f"{a} vs {b}", use_container_width=True):
            st.session_state["comp_a"] = a
            st.session_state["comp_b"] = b
            st.session_state["comp_c"] = c

    treatment_a = st.session_state.get("comp_a", treatment_a)
    treatment_b = st.session_state.get("comp_b", treatment_b)
    condition = st.session_state.get("comp_c", condition)

    if st.button("⚖️ Compare", type="primary", use_container_width=True):
        if treatment_a and treatment_b:
            with st.spinner(f"Comparing {treatment_a} vs {treatment_b}..."):
                from tools.medical_tools import compare_treatments_tool
                result = compare_treatments_tool.invoke(f"{treatment_a} {treatment_b} {condition}")
                st.markdown('<div class="response-box">', unsafe_allow_html=True)
                st.markdown(f"## {treatment_a} vs {treatment_b}")
                if condition:
                    st.markdown(f"*For: {condition}*")
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Save Comparison to Memory"):
                        st.session_state.agent.memory.save_entry(
                            researcher_id=st.session_state.researcher_id,
                            project_id=st.session_state.project_id,
                            disease_focus=condition or "general",
                            query=f"Compare {treatment_a} vs {treatment_b} for {condition}",
                            summary=result[:1000],
                            query_type="comparison",
                            approved=True,
                        )
                        st.success("Saved!")
                with col2:
                    if st.button("📋 Add to Report"):
                        st.session_state.report_sections.append({
                            "heading": f"Comparison: {treatment_a} vs {treatment_b}",
                            "content": result,
                        })
                        st.success("Added to report!")
        else:
            st.error("Please enter both treatments.")


# ─────────────────────────────────────────────
# PAGE 3: Memory Bank
# ─────────────────────────────────────────────
elif page == "💾 Memory Bank":
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0">💾 Long-Term Research Memory</h2>
        <p style="margin:4px 0 0">Browse and search past approved research findings across all sessions</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.agent_ready:
        st.warning("⚠️ Please initialize the agent first.")
        st.stop()

    memory = st.session_state.agent.memory
    all_entries = memory.get_all()

    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Total Memories", len(all_entries))
    col2.metric("✅ Approved", sum(1 for e in all_entries if e.get("approved")))
    col3.metric("👤 My Memories", sum(1 for e in all_entries if e.get("researcher_id") == st.session_state.researcher_id))

    st.divider()

    search_term = st.text_input("🔍 Search memories:", placeholder="e.g., diabetes, vaccine, cancer")
    filter_type = st.selectbox("Filter by type:", ["All", "summarization", "comparison", "clinical_qa", "trial_search", "general"])

    entries = memory.search(search_term) if search_term else all_entries
    if filter_type != "All":
        entries = [e for e in entries if e.get("query_type") == filter_type]

    if not entries:
        st.info("No memories found. Approve research responses to save them here.")
    else:
        for e in reversed(entries):
            with st.expander(f"📄 [{e['query_type']}] {e['query'][:70]}..."):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Researcher:** {e['researcher_id']} | **Project:** {e['project_id']}")
                    st.markdown(f"**Disease Focus:** {e['disease_focus']} | **Saved:** {e['timestamp'][:10]}")
                    st.markdown(e["summary"][:500] + "...")
                with col2:
                    st.markdown(f"**ID:** `{e['entry_id']}`")
                    status = "✅ Approved" if e.get("approved") else "⏳ Pending"
                    st.markdown(status)
                    if st.button("🗑️ Delete", key=f"del_{e['entry_id']}"):
                        memory.delete_entry(e['entry_id'])
                        st.rerun()
                    if st.button("📋 Add to Report", key=f"rep_{e['entry_id']}"):
                        st.session_state.report_sections.append({
                            "heading": e["query"][:60],
                            "content": e["summary"],
                        })
                        st.success("Added!")


# ─────────────────────────────────────────────
# PAGE 4: Report Builder
# ─────────────────────────────────────────────
elif page == "📋 Report Builder":
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0">📋 Research Report Builder</h2>
        <p style="margin:4px 0 0">Compile approved findings into a downloadable research report</p>
    </div>
    """, unsafe_allow_html=True)

    report_title = st.text_input("Report Title:", value="MediSyn Healthcare Research Report")

    sections = st.session_state.report_sections
    if not sections:
        st.info("No sections yet. Add research responses or memory entries using the 'Add to Report' buttons.")
    else:
        st.markdown(f"**{len(sections)} section(s) in report:**")
        for i, sec in enumerate(sections):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{i+1}.** {sec['heading']}")
            with col2:
                if st.button("Remove", key=f"rm_{i}"):
                    st.session_state.report_sections.pop(i)
                    st.rerun()

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            # Download as TXT
            from tools.report_generator import generate_text_report
            txt_report = generate_text_report(
                report_title, sections,
                st.session_state.researcher_id,
                st.session_state.project_id,
            )
            st.download_button(
                "📥 Download as TXT",
                data=txt_report,
                file_name=f"medisyn_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col2:
            # Download as PDF
            from tools.report_generator import generate_pdf_report
            pdf_bytes = generate_pdf_report(
                report_title, sections,
                st.session_state.researcher_id,
                st.session_state.project_id,
            )
            st.download_button(
                "📥 Download as PDF",
                data=pdf_bytes,
                file_name=f"medisyn_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with col3:
            if st.button("🗑️ Clear Report", use_container_width=True):
                st.session_state.report_sections = []
                st.rerun()

        # Preview
        st.divider()
        st.markdown("### 📄 Report Preview")
        for i, sec in enumerate(sections):
            with st.expander(f"Section {i+1}: {sec['heading']}"):
                st.markdown(sec["content"][:600] + "...")


# ─────────────────────────────────────────────
# PAGE 5: Settings
# ─────────────────────────────────────────────
elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Configuration")

    st.markdown("### 🧠 Memory Settings")
    limit = st.slider("Short-term memory limit (queries per session):", 3, 10, 7)
    st.markdown(f"Current: **{limit}** queries held in active session")

    st.markdown("### 📊 Session Statistics")
    if st.session_state.agent_ready:
        agent = st.session_state.agent
        col1, col2, col3 = st.columns(3)
        col1.metric("Session Queries", len(st.session_state.chat_history))
        col2.metric("Long-term Memories", agent.memory.count())
        col3.metric("Report Sections", len(st.session_state.report_sections))
    else:
        st.info("Initialize agent to see statistics.")

    st.markdown("### 🔧 About")
    st.markdown("""
    **MediSyn Labs AI Research Assistant**
    - **Framework**: AutoGen (Microsoft) + LangGraph (LangChain)
    - **LLM**: Google Gemini 1.5 Flash (free tier)
    - **Memory**: Short-term (session) + Long-term (JSON persistence)
    - **Tools**: DuckDuckGo Search, PubMed API (free), Gemini
    - **HITL**: Human-in-the-Loop approval before saving findings
    """)
