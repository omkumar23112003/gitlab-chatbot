"""
app.py  —  GitBot Streamlit UI
Run: streamlit run app.py
"""

import os
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitBot — GitLab Knowledge Assistant",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #161b22; }
.hero-title {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(120deg, #FC6D26 0%, #6B4FBB 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub { color: #8b949e; font-size: 1rem; margin-top: 2px; margin-bottom: 20px; }
.source-wrap { margin-top: 8px; }
.source-chip {
    display: inline-block; background: #21262d; border: 1px solid #30363d;
    border-radius: 20px; padding: 3px 12px; font-size: 0.74rem;
    color: #58a6ff; margin: 3px 3px 0 0; text-decoration: none;
}
.source-chip:hover { border-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_sources(sources: list, chunks: list):
    """Render clickable source chips."""
    if not sources:
        return
    chips = ""
    for url in sources[:6]:
        label = url.rstrip("/").split("/")[-1] or url.rstrip("/").split("/")[-2] or "page"
        label = label.replace("-", " ").title()[:30]
        chips += f'<a href="{url}" target="_blank" class="source-chip">🔗 {label}</a>'
    st.markdown(f'<div class="source-wrap">{chips}</div>', unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "rag"            not in st.session_state: st.session_state.rag            = None
if "pending_q"      not in st.session_state: st.session_state.pending_q      = None
if "total_queries"  not in st.session_state: st.session_state.total_queries  = 0
if "groq_key_cache" not in st.session_state: st.session_state.groq_key_cache = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦊 GitBot")
    st.caption("GitLab Handbook & Direction AI Assistant")
    st.divider()

    st.markdown("### 🔑 Groq API Key")
    default_key = (
        st.session_state.groq_key_cache
        or os.environ.get("GROQ_API_KEY", "")
        or st.secrets.get("GROQ_API_KEY", "")  # Streamlit Cloud
    )
    groq_key = st.text_input(
        "Enter your key",
        type="password",
        placeholder="gsk_...",
        help="100% FREE at https://console.groq.com — no credit card!",
        value=default_key,
    )
    if groq_key:
        st.session_state.groq_key_cache = groq_key
    st.caption("[Get free Groq key →](https://console.groq.com)")

    st.divider()
    st.markdown("### 🤖 Stack")
    st.markdown("""
| Component | Tech |
|-----------|------|
| **LLM** | Llama 3.3 70B |
| **Provider** | Groq (Free) |
| **Embeddings** | all-MiniLM-L6-v2 |
| **Vector DB** | FAISS (local) |
| **Reranking** | MMR |
""")

    st.divider()
    st.markdown("### ⚡ Quick Questions")
    quick_qs = [
        "What are GitLab's core values?",
        "How does GitLab handle remote work?",
        "What is GitLab's AI product direction?",
        "How does GitLab approach hiring?",
        "What is GitLab's DevSecOps strategy?",
        "How does GitLab define transparency?",
    ]
    for q in quick_qs:
        if st.button(q, use_container_width=True, key=f"qq_{hash(q)}"):
            st.session_state.pending_q = q

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages      = []
            st.session_state.chat_history  = []
            st.session_state.total_queries = 0
            st.rerun()
    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            st.session_state.rag = None
            st.rerun()


# ── Load RAG ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_cached(api_key: str):
    from rag_engine import GitLabRAG
    return GitLabRAG(api_key=api_key)


def ensure_rag() -> bool:
    key = st.session_state.groq_key_cache
    if not key:
        return False
    if st.session_state.rag is None:
        try:
            with st.spinner("⚙️ Loading knowledge base (first load ~20s)…"):
                st.session_state.rag = load_rag_cached(key)
            st.toast("✅ Knowledge base loaded!", icon="📚")
        except FileNotFoundError as e:
            st.error(
                f"**Knowledge base not found.**\n\n```\n{e}\n```\n\n"
                "Run these first:\n```bash\npython scraper.py\npython embed.py\n```"
            )
            return False
        except ValueError as e:
            st.error(str(e))
            return False
    return True


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🦊 GitBot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">AI-powered Q&A over GitLab\'s Handbook & Direction — '
    'Groq · FAISS · sentence-transformers · Free</p>',
    unsafe_allow_html=True,
)

rag_ready = ensure_rag()

if rag_ready:
    rag = st.session_state.rag
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Indexed Vectors", f"{rag.index_size:,}")
    c2.metric("💬 Messages", len(st.session_state.messages))
    c3.metric("🔍 Queries", st.session_state.total_queries)
    c4.metric("🚀 Model", "Llama 3.3 70B")
else:
    st.info("👈 Enter your **free Groq API key** in the sidebar, then ask anything.")

st.divider()


# ── Welcome ───────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🦊"):
        st.markdown("""
**Hey! I'm GitBot 👋**

I can answer questions about GitLab's:
- 📖 **Handbook** — values, culture, processes, remote work, hiring
- 🔭 **Direction** — product strategy, roadmap, DevSecOps vision, AI plans

I use **semantic search** (FAISS + embeddings) over real GitLab docs, so answers are grounded in actual content — not hallucinated.

**Get started:** Enter your free Groq API key in the sidebar, then ask away!
""")


# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🦊" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"], msg.get("chunks", []))


# ── Input ─────────────────────────────────────────────────────────────────────
prompt = st.session_state.pending_q
if prompt:
    st.session_state.pending_q = None
else:
    prompt = st.chat_input(
        "Ask anything about GitLab's Handbook or Direction…",
        disabled=not rag_ready,
    )

if prompt:
    if not rag_ready:
        st.warning("⚠️ Enter your Groq API key in the sidebar first.")
        st.stop()

    st.session_state.total_queries += 1

    # Show user bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant", avatar="🦊"):
        with st.spinner("🔍 Searching knowledge base…"):
            try:
                result  = st.session_state.rag.ask(
                    prompt,
                    chat_history=st.session_state.chat_history,
                )
                answer  = result["answer"]
                sources = result["sources"]
                chunks  = result["chunks"]

                st.markdown(answer)
                render_sources(sources, chunks)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "chunks": chunks,
                })
                st.session_state.chat_history.extend([
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": answer},
                ])

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
