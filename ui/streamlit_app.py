"""Streamlit front end for the Regulatory Q&A API.

    streamlit run ui/streamlit_app.py        # expects the API at $RQA_API_URL (default localhost:8000)
"""
import json
import os
import re

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("RQA_API_URL", "http://localhost:8000")
LANGFUSE = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
LANGFUSE_PROJECT = "regulatory-qa"

EXAMPLES = [
    "What data-integrity problems did FDA find at Curia?",
    "Which companies were cited under 21 CFR Part 211, and for what?",
    "What pathogens were found in Raaw Energy's dog food?",
    "Which regulations are cited most often, and by which companies?",
    "What did FDA say about OOS investigations at firms cited under 21 CFR 211.192?",
    "How much was Bentley Laboratories fined?",
]

STATUS = {
    "answered": ("✅ Answered", "green", "Every claim is backed by a cited passage."),
    "flagged": ("⚠️ Flagged", "orange", "Answer shown, but some checks failed (see flags)."),
    "refused": ("🚫 Not in the letters", "gray", "The indexed letters don't support an answer."),
    "rejected": ("⛔ Rejected", "red", "The draft had no valid citations, so it isn't shown."),
}

st.set_page_config(page_title="Regulatory Q&A", page_icon="📑", layout="wide")


# --- API helpers ---------------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict:
    try:
        r = httpx.get(f"{API}/health", timeout=5)
        return r.json() if r.status_code == 200 else {"status": "degraded", **r.json().get("detail", {})}
    except httpx.HTTPError as e:
        return {"status": "down", "error": type(e).__name__}


@st.cache_data(ttl=300, show_spinner=False)
def documents() -> list[dict]:
    return httpx.get(f"{API}/documents", timeout=10).json()


@st.cache_data(ttl=300, show_spinner=False)
def document(doc_id: str) -> dict:
    return httpx.get(f"{API}/documents/{doc_id}", timeout=10).json()


def stream_ask(payload: dict):
    """Yield (event, data) pairs from the SSE stream of POST /ask."""
    with httpx.stream("POST", f"{API}/ask", json={**payload, "stream": True}, timeout=300) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[6:])
                event = None


# --- rendering -----------------------------------------------------------------------------------

def _anchor(qid: int, label: str) -> str:
    return f"q{qid}-{label.lower()}"


def linkify(answer: str, qid: int) -> str:
    """[S3] / [S1, S4] -> links that jump to the matching source below the answer."""
    def repl(m: re.Match) -> str:
        labels = re.findall(r"S\d+", m.group(0))
        return "[" + ", ".join(f"[{lab}](#{_anchor(qid, lab)})" for lab in labels) + "]"
    return re.sub(r"\[S\d+(?:,\s*S\d+)*\]", repl, answer)


def render_result(qid: int, question: str, res: dict) -> None:
    label, color, help_text = STATUS.get(res["status"], (res["status"], "gray", ""))
    st.markdown(f"#### {question}")
    st.markdown(f":{color}-badge[{label}]  <small>{help_text}</small>", unsafe_allow_html=True)
    if res.get("cached"):
        st.caption("⚡ Served from the answer cache: no LLM calls.")
    if res.get("flags"):
        st.caption("Flags: " + " · ".join(f"`{f}`" for f in res["flags"]))
    st.markdown(linkify(res["answer"], qid))

    if res.get("citations"):
        st.markdown("**Sources cited**")
        for c in res["citations"]:
            st.markdown(f'<div id="{_anchor(qid, c["label"])}"></div>', unsafe_allow_html=True)
            # Many passages share a section heading, so the title previews the passage itself.
            preview = re.sub(r"\s+", " ", c["snippet"])[:90]
            title = f'{c["label"]} · {c["company"]} ({c["issue_date"]}) · “{preview}…”'
            with st.expander(title):
                st.caption(c["section"])
                st.write(c["snippet"])
                # New browser tab: a same-tab link would start a new session and drop the history.
                letter_view = f'?doc={c["doc_id"]}&chunk={c["chunk_id"]}'
                links = [f'<a href="{letter_view}" target="_blank">Open in letter view ↗</a>']
                if c.get("url"):
                    links.append(f'<a href="{c["url"]}" target="_blank">FDA letter ↗</a>')
                st.markdown(" · ".join(links), unsafe_allow_html=True)

    with st.expander("How this was answered"):
        plan = res.get("plan") or {}
        cols = st.columns(4)
        cols[0].metric("Route", res.get("route", "–"))
        cols[1].metric("Total time", f'{res.get("timings_ms", {}).get("total", 0) / 1000:.1f}s')
        usage = res.get("usage") or {}
        cols[2].metric("Answer tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
        cols[3].metric("Cost (list price)", f'${res.get("cost_usd", 0):.4f}')
        if res.get("trace_id"):
            st.markdown(f'<a href="{LANGFUSE}/project/{LANGFUSE_PROJECT}/traces/{res["trace_id"]}" '
                        'target="_blank">View trace in Langfuse ↗</a>', unsafe_allow_html=True)
        st.caption(f'Search query: “{plan.get("search_query", "")}”'
                   + (f' · company filter: {plan["company"]}' if plan.get("company") else ""))
        graph = res.get("graph")
        if graph:
            st.markdown(f'**Graph query** `{graph["template"]}` '
                        f'{ {k: v for k, v in graph["params"].items() if v not in (None, True, 25)} }')
            if graph.get("error"):
                st.error(graph["error"])
            elif graph["rows"]:
                df = pd.DataFrame(graph["rows"]).drop(columns=["chunk_ids", "observation_id", "doc_id"],
                                                      errors="ignore")
                st.dataframe(df, hide_index=True, use_container_width=True)
        if res.get("sources"):
            st.markdown("**Passages given to the model**")
            st.dataframe(pd.DataFrame([{
                "label": s["label"], "origin": s["origin"], "company": s["company"],
                "section": s["section"][:50], "rerank": s.get("rerank_score"), "chunk": s["chunk_id"],
            } for s in res["sources"]]), hide_index=True, use_container_width=True)
        st.caption("Timings (ms): " + ", ".join(f"{k} {v}" for k, v in res.get("timings_ms", {}).items()))


# --- sidebar -------------------------------------------------------------------------------------

with st.sidebar:
    st.title("📑 Regulatory Q&A")
    st.caption("Questions over FDA warning letters, answered only from cited passages.")
    h = health()
    if h["status"] == "ok":
        st.success("API connected", icon="🟢")
        for name, val in h.get("checks", {}).items():
            st.caption(f"{name}: {val}")
    else:
        st.error(f"API {h['status']} at {API}", icon="🔴")
        st.caption("Start it with `uvicorn app.api.main:app`")

    st.subheader("Retrieval settings")
    mode = st.selectbox("Search mode", ["hybrid", "vector", "keyword"],
                        help="Hybrid fuses exact-citation, vector and keyword search.")
    use_graph = st.toggle("Use knowledge graph", value=True,
                          help="Let the planner run a whitelisted graph query for list/filter/count questions.")
    top_k = st.slider("Passages from text search", 3, 15, 8)

    st.subheader("Try a question")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex


# --- tabs ----------------------------------------------------------------------------------------

params = st.query_params
ask_tab, letters_tab = st.tabs(["Ask", "Letters"], default="Letters" if params.get("doc") else "Ask")

with ask_tab:
    st.session_state.setdefault("history", [])
    typed = st.chat_input("Ask about the indexed FDA warning letters…")
    question = typed or st.session_state.pop("pending", None)

    if question:
        qid = len(st.session_state.history)
        progress = st.empty()
        live = st.empty()
        text = ""
        result = None
        sources: list[dict] = []
        try:
            with progress.status("Planning and retrieving…", expanded=False) as status:
                for event, data in stream_ask({"question": question, "mode": mode,
                                               "use_graph": use_graph, "top_k": top_k}):
                    if event == "plan":
                        g = data.get("graph")
                        status.update(label=f'Route: {data["route"]}'
                                      + (f' · graph: {g["template"]} ({len(g["rows"])} rows)' if g else ""))
                    elif event == "sources":
                        sources = data
                        status.update(label=f"Writing answer from {len(data)} passages…")
                    elif event == "token":
                        text += data["text"]
                        live.markdown(text + " ▌")
                    elif event == "error":
                        icon = "⏳" if data.get("code") in ("quota_exhausted", "rate_limited") else "⚠️"
                        st.error(data["message"], icon=icon)
                        status.update(label="Couldn't generate an answer", state="error")
                    elif event == "done":
                        result = data
                if result:
                    status.update(label="Done", state="complete")
        except httpx.HTTPError as e:
            st.error(f"Request failed: {e}")
        if result:
            progress.empty()
        live.empty()  # the checked answer replaces the raw stream (citations may be stripped or rejected)
        if result:
            # the done event omits sources (sent earlier in the stream); keep them for the debug panel
            st.session_state.history.append((question, {**result, "sources": sources}))

    for qid, (q, res) in reversed(list(enumerate(st.session_state.history))):
        with st.container(border=True):
            render_result(qid, q, res)
    if not st.session_state.history:
        st.info("Ask a question below, or pick an example from the sidebar.")

with letters_tab:
    docs = documents() if health()["status"] == "ok" else []
    if not docs:
        st.warning("No letters available.")
    else:
        ids = [d["id"] for d in docs]
        default = ids.index(params["doc"]) if params.get("doc") in ids else 0
        names = {d["id"]: f'{d["company"]} — {d["issue_date"]}' for d in docs}
        doc_id = st.selectbox("Warning letter", ids, index=default, format_func=names.get)
        d = document(doc_id)
        st.subheader(d["company"])
        st.caption(f'{d["subject"]} · issued {d["issue_date"]} · {d["issuing_office"]} · {d["location"]}')
        st.markdown(f'[Read on fda.gov ↗]({d["url"]})')

        focus = params.get("chunk")
        cited = next((c for c in d["chunks"] if c["id"] == focus), None)
        if cited:  # opened from a citation: show that passage first
            with st.container(border=True):
                st.markdown(f'📌 **Cited passage** · {cited["section"]}')
                st.write(cited["text"])

        st.markdown(f'**Observations ({len(d["observations"])})**')
        st.dataframe(pd.DataFrame([{
            "#": o["number"], "title": o["title"], "regulations": ", ".join(o["regulations"]),
            "topics": ", ".join(o["topics"]), "summary": o["summary"],
        } for o in d["observations"]]), hide_index=True, use_container_width=True)

        obs_title = {o["observation_id"]: o["title"] for o in d["observations"]}
        st.markdown(f'**Passages ({len(d["chunks"])})**')
        for c in d["chunks"]:
            tag = f' · {obs_title[c["observation_id"]]}' if c["observation_id"] in obs_title else ""
            with st.expander(f'{c["ordinal"]:03d} · {c["section"][:60]}{tag}', expanded=c["id"] == focus):
                if c["id"] == focus:
                    st.success("Cited passage", icon="📌")
                st.write(c["text"])
                st.caption(c["id"])
