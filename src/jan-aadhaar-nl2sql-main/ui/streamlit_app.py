"""
ui/streamlit_app.py — Streamlit front-end for the Jan-Aadhaar NL2SQL pipeline.
Connects to the FastAPI backend at http://localhost:8000.
"""
import httpx
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Jan-Aadhaar NL2SQL",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #0f2139 100%);
    color: #e2e8f0;
}

/* ── Header ── */
.header-wrap {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.8rem;
}
.header-wrap h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(100deg, #818cf8, #60a5fa, #34d399, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.header-wrap .tagline {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 400;
}

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.22rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-right: 0.35rem;
}
.badge-cache   { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.35); }
.badge-llm     { background: rgba(129,140,248,0.12); color: #818cf8; border: 1px solid rgba(129,140,248,0.35); }
.badge-warn    { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.35); }
.badge-hint    { background: rgba(96,165,250,0.10); color: #60a5fa; border: 1px solid rgba(96,165,250,0.30); }
.badge-norm    { background: rgba(168,85,247,0.10); color: #c084fc; border: 1px solid rgba(168,85,247,0.30); }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 0.85rem; flex-wrap: wrap; margin: 1rem 0 1.2rem 0; }
.metric-card {
    flex: 1; min-width: 110px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(129,140,248,0.35); }
.metric-card .m-label {
    font-size: 0.68rem; color: #475569;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 0.25rem;
}
.metric-card .m-value { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; }
.metric-card .m-unit { font-size: 0.72rem; color: #64748b; margin-left: 2px; }

/* ── SQL block ── */
.sql-block {
    background: rgba(8,8,25,0.85);
    border: 1px solid rgba(129,140,248,0.25);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 0.88rem;
    color: #c4b5fd;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
    margin-bottom: 1.2rem;
    position: relative;
}
.sql-block::before {
    content: "SQL";
    position: absolute; top: 0.5rem; right: 0.9rem;
    font-size: 0.65rem; color: #4c4f6b; font-weight: 600;
    letter-spacing: 0.1em; font-family: 'Inter', sans-serif;
}

/* ── Hint chips ── */
.hint-chip {
    display: inline-block;
    background: rgba(96,165,250,0.08);
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 6px;
    padding: 0.18rem 0.6rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.78rem;
    color: #93c5fd;
    margin: 0.15rem 0.2rem;
}

/* ── Pipeline trace ── */
.pipeline-step {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 0.8rem;
    background: rgba(255,255,255,0.025);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: #94a3b8;
}
.pipeline-step .step-name { color: #e2e8f0; font-weight: 500; min-width: 130px; }
.pipeline-step .step-ms { color: #64748b; font-size: 0.75rem; margin-left: auto; font-family: monospace; }

/* ── Input ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(129,140,248,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    line-height: 1.5 !important;
}
.stTextArea textarea:focus {
    border-color: rgba(129,140,248,0.6) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.1) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(8,8,25,0.97) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    padding: 0.35rem 0.8rem !important;
    text-align: left !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(129,140,248,0.12) !important;
    border-color: rgba(129,140,248,0.3) !important;
    color: #e2e8f0 !important;
    transform: none !important;
}

div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stExpander { border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)



# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <h1>🏛️ Jan-Aadhaar NL2SQL</h1>
    <p class="tagline">
        Ask in plain English &nbsp;·&nbsp; Instant SQL &nbsp;·&nbsp;
        RapidFuzz · ChromaDB · Ollama · DuckDB &nbsp;·&nbsp; 100% Local
    </p>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
default_q = st.session_state.get("q_input", "")
question = st.text_area(
    "question",
    value=default_q,
    placeholder="e.g. How many poor female members are there in rural Jaipur?",
    height=80,
    key="main_q",
    label_visibility="collapsed",
)

c1, c2, c3 = st.columns([2, 1, 6])
with c1:
    run_clicked = st.button("⚡ Run Query", use_container_width=True)
with c2:
    if st.button("✕", use_container_width=True):
        st.session_state["q_input"] = ""
        st.rerun()

# ── Query execution ───────────────────────────────────────────────────────────
if run_clicked and question.strip():
    with st.spinner("Processing through pipeline..."):
        try:
            resp = httpx.post(
                f"{API_BASE}/query",
                json={"question": question.strip()},
                timeout=360,
            )

            if resp.status_code == 200:
                data = resp.json()

                # ── Source badge + normalization notice ────────────────────────
                badge_row = ""
                if data["source"] == "cache":
                    badge_row += '<span class="badge badge-cache">⚡ Cache Hit</span>'
                else:
                    badge_row += '<span class="badge badge-llm">🤖 LLM Generated</span>'

                if data.get("correction_attempts", 0) > 0:
                    badge_row += f'<span class="badge badge-warn">🔁 {data["correction_attempts"]} correction(s)</span>'

                nq = data.get("normalized_question", "")
                if nq and nq != data["question"]:
                    badge_row += f'<span class="badge badge-norm">✏️ Normalized</span>'

                st.markdown(badge_row, unsafe_allow_html=True)

                # ── Normalization detail ───────────────────────────────────────
                if nq and nq != data["question"]:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:#94a3b8;margin:0.4rem 0 0.2rem 0;">'
                        f'Input normalized: <span style="color:#c084fc;font-family:monospace">{nq}</span></div>',
                        unsafe_allow_html=True,
                    )

                # ── Domain hints ───────────────────────────────────────────────
                hints = data.get("domain_hints", [])
                if hints:
                    chips = "".join(f'<span class="hint-chip">{h}</span>' for h in hints)
                    st.markdown(
                        f'<div style="margin:0.5rem 0 0.8rem 0;">'
                        f'<span class="badge badge-hint">🎯 Domain Hints</span> {chips}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Latency metrics ────────────────────────────────────────────
                lat = data["latency"]
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="m-label">Total</div>
                        <div class="m-value">{lat.get('total_ms',0):.0f}<span class="m-unit">ms</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">Normalize</div>
                        <div class="m-value">{lat.get('normalize_ms',0):.1f}<span class="m-unit">ms</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">Cache</div>
                        <div class="m-value">{lat.get('cache_lookup_ms',0):.0f}<span class="m-unit">ms</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">RAG</div>
                        <div class="m-value">{lat.get('rag_ms',0):.0f}<span class="m-unit">ms</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">LLM</div>
                        <div class="m-value">{lat.get('llm_generation_ms',0):.0f}<span class="m-unit">ms</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">DuckDB</div>
                        <div class="m-value">{lat.get('db_execution_ms',0):.0f}<span class="m-unit">ms</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">Rows</div>
                        <div class="m-value">{data['row_count']:,}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Pipeline Trace ──────────────────────────────────────────────
                steps = data.get("pipeline_steps", [])
                if steps:
                    with st.expander("🔍 Execution Path (Step-by-Step)", expanded=False):
                        trace_html = ""
                        for step in steps:
                            # Use icon based on status
                            if step["status"] == "hit": icon = "🎯"
                            elif step["status"] == "miss": icon = "❌"
                            elif step["status"] == "skipped": icon = "⏭️"
                            else: icon = "✅" # used
                            
                            opacity = "0.5" if step["status"] == "skipped" else "1.0"
                            
                            detail_html = f'<span style="color: #94a3b8; margin-left: 10px;">— {step["detail"]}</span>' if step.get("detail") else ""
                            
                            trace_html += f"""<div class="pipeline-step" style="opacity: {opacity}">
    <span style="font-size: 1.1rem; width: 24px;">{icon}</span>
    <span class="step-name">{step['name']}</span>
    {detail_html}
    <span class="step-ms">{step['ms']:.1f} ms</span>
</div>"""
                        st.markdown(trace_html, unsafe_allow_html=True)

                # ── SQL display ────────────────────────────────────────────────
                st.markdown("#### Generated SQL")
                st.markdown(
                    f'<div class="sql-block">{data["sql"]}</div>',
                    unsafe_allow_html=True,
                )

                # ── Results table ──────────────────────────────────────────────
                if data["rows"]:
                    trunc_note = " *(showing first 500)*" if data["truncated"] else ""
                    st.markdown(f"#### Results{trunc_note}")
                    df = pd.DataFrame(data["rows"])
                    st.dataframe(df, use_container_width=True, height=420)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Query returned no rows.")

                # ── RAG Selected Columns (expandable) ────────────────────────────────
                rag_cols = data.get("rag_columns", [])
                if rag_cols:
                    with st.expander("📚 RAG Selected Columns", expanded=False):
                        st.markdown("These schema columns were injected into the LLM prompt to provide dataset context:")
                        df_rag = pd.DataFrame(rag_cols)
                        st.dataframe(df_rag, use_container_width=True, hide_index=True)

            else:
                detail = resp.json().get("detail", resp.text)
                st.error(f"Error {resp.status_code}: {detail}")

        except httpx.ConnectError:
            st.error(
                "**Cannot reach the API** at `http://localhost:8000`.\n\n"
                "Start it with:\n```\npython main.py\n```"
            )
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

elif run_clicked:
    st.warning("Please enter a question.")


# ── Schema explorer ───────────────────────────────────────────────────────────
with st.expander("📋 Dataset Schema", expanded=False):
    try:
        schema = httpx.get(f"{API_BASE}/schema", timeout=5).json()
        rows = [
            {"Column": k, "Type": v["type"], "Description": v["description"]}
            for k, v in schema.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception:
        st.warning("API not reachable. Run `python main.py` first.")
