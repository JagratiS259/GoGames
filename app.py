"""
Company Data & RAG Platform
Streamlit · FAISS · Groq (llama-3.3-70b) · Tavily web search
"""

import os, json, textwrap
import numpy as np
import faiss
import streamlit as st
import requests
from groq import Groq
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Company RAG",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50
TOP_K         = 4
GROQ_MODEL    = "llama-3.3-70b-versatile"
PERSIST_FILE  = "companies.json"
TAVILY_URL    = "https://api.tavily.com/search"

SECTOR_OPTIONS = ["Technology","Finance","Healthcare","Retail",
                  "Manufacturing","Energy","Real Estate","Media","Consulting","Other"]
DOC_TYPES      = ["Financial","Strategy","HR","Compliance","Governance","Operations","Legal","Other"]
BADGE_COLORS   = {"Financial":"green","Strategy":"blue","HR":"orange",
                  "Compliance":"violet","Governance":"red","Legal":"violet","Other":"grey"}

# ── Data models ───────────────────────────────────────────────────────
@dataclass
class Document:
    name    : str
    doc_type: str
    date    : str
    content : str = ""

    def to_index_text(self) -> str:
        return f"[{self.doc_type} | {self.name} | {self.date}]\n{self.content}"

@dataclass
class Company:
    name       : str
    sector     : str
    founded    : int
    hq         : str
    revenue    : str
    employees  : int
    description: str
    documents  : List[Document] = field(default_factory=list)

    def structured_profile(self) -> str:
        return (
            f"Company : {self.name}\n"
            f"Sector  : {self.sector}\n"
            f"Founded : {self.founded}\n"
            f"HQ      : {self.hq}\n"
            f"Revenue : ${self.revenue} (annual)\n"
            f"Staff   : {self.employees:,}\n"
            f"About   : {self.description}"
        )

    def to_dict(self) -> dict:
        return dict(
            name=self.name, sector=self.sector, founded=self.founded,
            hq=self.hq, revenue=self.revenue, employees=self.employees,
            description=self.description,
            documents=[dict(name=d.name, doc_type=d.doc_type,
                            date=d.date, content=d.content)
                       for d in self.documents],
        )

# ── Cached embedder ───────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)

# ── Chunker ───────────────────────────────────────────────────────────
def chunk_text(text: str) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c]

# ── FAISS store ───────────────────────────────────────────────────────
class FAISSStore:
    def __init__(self, dim: int):
        self.index  = faiss.IndexFlatL2(dim)
        self.chunks : List[str]  = []
        self.meta   : List[dict] = []

    def add(self, doc: Document, embedder):
        chunks = chunk_text(doc.to_index_text())
        vecs   = embedder.encode(chunks, normalize_embeddings=True)
        self.index.add(np.array(vecs, dtype="float32"))
        for c in chunks:
            self.chunks.append(c)
            self.meta.append({"doc": doc.name, "type": doc.doc_type, "date": doc.date})

    def search(self, query: str, embedder, k: int = TOP_K) -> List[Tuple[str, dict, float]]:
        if self.index.ntotal == 0:
            return []
        q    = embedder.encode([query], normalize_embeddings=True)
        k    = min(k, self.index.ntotal)
        D, I = self.index.search(np.array(q, dtype="float32"), k)
        return [(self.chunks[i], self.meta[i], float(D[0][r]))
                for r, i in enumerate(I[0]) if i != -1]

# ── Tavily web search ─────────────────────────────────────────────────
def tavily_search(query: str, api_key: str, max_results: int = 4) -> List[dict]:
    """Returns list of {title, url, content} from Tavily."""
    try:
        resp = requests.post(
            TAVILY_URL,
            json={"api_key": api_key, "query": query,
                  "max_results": max_results, "search_depth": "basic"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        return [{"title": "Search error", "url": "", "content": str(e)}]

# ── Session state ─────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        companies={}, indexes={}, histories={},
        active=None, tab="Data",
        groq_key=os.environ.get("GROQ_API_KEY", ""),
        tavily_key=os.environ.get("TAVILY_API_KEY", ""),
        web_search=False,
        show_add_company=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def add_company(company: Company):
    embedder = load_embedder()
    key = company.name.lower()
    st.session_state.companies[key]  = company
    st.session_state.indexes[key]    = FAISSStore(embedder.get_sentence_embedding_dimension())
    st.session_state.histories[key]  = []
    st.session_state.indexes[key].add(
        Document("Company Profile","Profile","2025", company.structured_profile()), embedder
    )
    for doc in company.documents:
        st.session_state.indexes[key].add(doc, embedder)

def add_document_to(company_name: str, doc: Document):
    embedder = load_embedder()
    key = company_name.lower()
    st.session_state.companies[key].documents.append(doc)
    st.session_state.indexes[key].add(doc, embedder)

def save_companies():
    data = {k: c.to_dict() for k, c in st.session_state.companies.items()}
    with open(PERSIST_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_companies_from_disk():
    if not os.path.exists(PERSIST_FILE):
        return
    with open(PERSIST_FILE) as f:
        data = json.load(f)
    for _, c in data.items():
        company = Company(
            name=c["name"], sector=c["sector"], founded=c["founded"],
            hq=c["hq"], revenue=c["revenue"], employees=c["employees"],
            description=c["description"],
            documents=[Document(**d) for d in c.get("documents", [])],
        )
        add_company(company)

# ── Prompt templates ──────────────────────────────────────────────────
SYSTEM_DOCS_ONLY = (
    "You are a company research assistant.\n"
    "Answer using ONLY the company profile and retrieved document chunks below.\n"
    "Cite the document name when referencing specific data.\n"
    "If the answer is not in the context, say so clearly.\n\n"
    "--- COMPANY PROFILE ---\n{profile}\n\n"
    "--- RETRIEVED DOCUMENT CHUNKS ---\n{context}"
)

SYSTEM_WITH_WEB = (
    "You are a company research assistant with access to internal documents AND live web search results.\n"
    "Prioritise internal documents for company-specific data.\n"
    "Use web results for recent news, market context, or anything not in the documents.\n"
    "Always cite your source (document name or URL).\n\n"
    "--- COMPANY PROFILE ---\n{profile}\n\n"
    "--- RETRIEVED DOCUMENT CHUNKS ---\n{context}\n\n"
    "--- LIVE WEB SEARCH RESULTS ---\n{web}"
)

SAMPLE_COMPANIES = [
    Company(
        name="Nexus Analytics", sector="Technology", founded=2018,
        hq="Austin, TX", revenue="12,400,000", employees=340,
        description="AI-powered business intelligence platform for mid-market companies.",
        documents=[
            Document("Q3 2024 Earnings","Financial","Oct 2024",
                "Revenue Q3 2024: $3.1M (+22% YoY). Gross margin: 71%. "
                "Operating loss: -$420K. ARR: $12.4M. Net new ARR: $1.8M. "
                "Top segment: manufacturing (38%). Cash: $8.2M, 18-month runway."),
            Document("Product Roadmap 2025","Strategy","Dec 2024",
                "H1 2025: predictive churn module + mobile dashboard. "
                "H2 2025: EMEA expansion, Nexus Pro at $2,500/month. "
                "Key hires: VP Sales, 3 engineers, 2 CS managers. Target ARR: $20M."),
            Document("Employee Handbook","HR","Jan 2024",
                "Remote-first. Core hours 10am-3pm local. Unlimited PTO (15-day min). "
                "$2,000 annual learning budget. 4-year equity vesting, 1-year cliff."),
        ]
    ),
    Company(
        name="GreenVault Capital", sector="Finance", founded=2015,
        hq="New York, NY", revenue="87,000,000", employees=120,
        description="ESG-focused private equity fund investing in sustainable infrastructure.",
        documents=[
            Document("Fund Performance Summary","Financial","Sep 2024",
                "Fund III AUM: $1.2B. Net IRR: 18.4%. 14 active investments. "
                "Largest: SunBridge Energy (32%). 2 exits in 2024 at 2.8x MOIC. 42 LPs."),
            Document("ESG Criteria Guidelines","Compliance","Mar 2024",
                "Carbon-reduction: 40% by 2030. Board diversity: ≥30% women. "
                "Annual ESG audit required. Exclusion: fossil fuels, single-use plastics."),
        ]
    ),
]

# ══════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════
init_state()

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 260px; max-width: 260px; }
.company-btn > button { text-align: left !important; }
div[data-testid="metric-container"] { background:#f5f7fa; border-radius:8px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Company RAG")
    st.caption("FAISS · Groq · Tavily")

    with st.expander("⚙️ API Keys", expanded=not (st.session_state.groq_key and st.session_state.tavily_key)):
        g = st.text_input("Groq API Key", value=st.session_state.groq_key,
                          type="password", placeholder="gsk_...")
        t = st.text_input("Tavily API Key", value=st.session_state.tavily_key,
                          type="password", placeholder="tvly-...",
                          help="Optional — enables live web search")
        if g != st.session_state.groq_key: st.session_state.groq_key = g
        if t != st.session_state.tavily_key: st.session_state.tavily_key = t
        st.caption("[Get Groq key](https://console.groq.com) · [Get Tavily key](https://tavily.com)")

    st.divider()

    if not st.session_state.companies:
        if st.button("📦 Load sample companies", use_container_width=True):
            for sc in SAMPLE_COMPANIES:
                add_company(sc)
            st.session_state.active = list(st.session_state.companies.keys())[0]
            st.rerun()
        if os.path.exists(PERSIST_FILE):
            if st.button("📂 Load saved", use_container_width=True):
                load_companies_from_disk()
                if st.session_state.companies:
                    st.session_state.active = list(st.session_state.companies.keys())[0]
                st.rerun()
    else:
        st.subheader("Companies")
        for key, company in st.session_state.companies.items():
            is_active = st.session_state.active == key
            label = f"**{company.name}**  \n{company.sector} · {len(company.documents)} docs"
            if st.button(label, key=f"sel_{key}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active = key
                st.session_state.tab    = "Data"
                st.rerun()

        st.divider()
        if st.button("➕ Add company", use_container_width=True):
            st.session_state.show_add_company = True
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", use_container_width=True):
                save_companies(); st.success("Saved!")
        with c2:
            if st.button("📂 Load", use_container_width=True):
                load_companies_from_disk(); st.rerun()

# ── Add Company form ──────────────────────────────────────────────────
if st.session_state.show_add_company:
    with st.container(border=True):
        st.subheader("➕ New Company")
        c1, c2 = st.columns(2)
        with c1:
            nn = st.text_input("Name *", key="nn")
            ns = st.selectbox("Sector", SECTOR_OPTIONS, key="ns")
            nf = st.number_input("Founded", 1800, 2025, 2020, key="nf")
        with c2:
            nh = st.text_input("HQ", key="nh")
            nr = st.text_input("Revenue (USD)", placeholder="12,400,000", key="nr")
            ne = st.number_input("Employees", 0, 1_000_000, 100, key="ne")
        nd = st.text_area("Description", key="nd")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Add", type="primary", use_container_width=True):
                if nn:
                    co = Company(name=nn, sector=ns, founded=int(nf), hq=nh or "Unknown",
                                 revenue=nr or "0", employees=int(ne), description=nd or "—")
                    add_company(co)
                    st.session_state.active = co.name.lower()
                    st.session_state.show_add_company = False
                    st.rerun()
                else:
                    st.error("Name required.")
        with b2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_add_company = False; st.rerun()

# ── Empty state ───────────────────────────────────────────────────────
if not st.session_state.companies:
    st.markdown("## 👋 Welcome")
    st.info("Add your API keys in the sidebar, then load sample companies or add your own.")
    st.markdown("""
**Pipeline:**
1. Add companies + documents → chunked & embedded → stored in **FAISS**
2. Ask a question → top-K chunks retrieved by cosine similarity
3. Optional **Tavily** web search appends live results
4. **Groq** streams a grounded answer in real-time
    """)
    st.stop()

active_key = st.session_state.active
if not active_key or active_key not in st.session_state.companies:
    st.info("← Select a company from the sidebar.")
    st.stop()

company = st.session_state.companies[active_key]
store   = st.session_state.indexes[active_key]

# ── Header ────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.title(f"🏢 {company.name}")
    st.caption(f"{company.sector} · {company.hq} · Founded {company.founded}")
with hcol2:
    tab_choice = st.radio("", ["Data", "Ask AI ✦"], horizontal=True,
                          key="tab_radio",
                          index=0 if st.session_state.tab == "Data" else 1)
    st.session_state.tab = tab_choice

st.divider()

# ══════════════════════════════════════════════════════════════════════
# DATA TAB
# ══════════════════════════════════════════════════════════════════════
if st.session_state.tab == "Data":

    # Metrics
    try:
        rev_n   = int(company.revenue.replace(",",""))
        rev_fmt = f"${rev_n/1e6:.1f}M" if rev_n >= 1e6 else f"${rev_n:,}"
        rpe     = f"${rev_n // company.employees:,}" if company.employees else "—"
    except Exception:
        rev_fmt, rpe = f"${company.revenue}", "—"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Revenue",          rev_fmt)
    m2.metric("Employees",        f"{company.employees:,}")
    m3.metric("Founded",          company.founded)
    m4.metric("Rev / Employee",   rpe)
    m5.metric("FAISS Vectors",    store.index.ntotal)

    st.markdown(f"**About:** {company.description}")
    st.divider()

    # Documents
    st.subheader(f"📄 Documents ({len(company.documents)})")
    if not company.documents:
        st.caption("No documents yet — add one below.")
    for doc in company.documents:
        color = BADGE_COLORS.get(doc.doc_type, "grey")
        with st.expander(f":{color}[{doc.doc_type}]  **{doc.name}** — {doc.date}"):
            st.text(doc.content or "No content.")

    # Add document form
    st.subheader("➕ Add Document")
    with st.form("add_doc_form", clear_on_submit=True):
        d1, d2, d3 = st.columns(3)
        dname    = d1.text_input("Name *")
        dtype    = d2.selectbox("Type", DOC_TYPES)
        ddate    = d3.text_input("Date", placeholder="Jan 2025")
        dcontent = st.text_area("Content (indexed for RAG retrieval)", height=120)
        if st.form_submit_button("Add & Index", type="primary"):
            if dname:
                add_document_to(active_key, Document(dname, dtype, ddate or "2025", dcontent))
                st.success(f"✅ '{dname}' indexed — {store.index.ntotal} total vectors")
                st.rerun()
            else:
                st.error("Name required.")

# ══════════════════════════════════════════════════════════════════════
# ASK AI TAB
# ══════════════════════════════════════════════════════════════════════
else:
    if not st.session_state.groq_key:
        st.warning("⚠️ Enter your Groq API key in the sidebar.")
        st.stop()

    history = st.session_state.histories[active_key]

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        web_on = st.toggle(
            "🌐 Tavily web search",
            value=st.session_state.web_search,
            disabled=not st.session_state.tavily_key,
            help="Enriches answers with live web results (requires Tavily key)",
        )
        st.session_state.web_search = web_on
    with ctrl2:
        if st.session_state.tavily_key and web_on:
            st.success("Web search ON", icon="🌐")
        elif not st.session_state.tavily_key:
            st.caption("Add Tavily key to enable web search")
    with ctrl3:
        if history and st.button("🗑️ Clear", use_container_width=True):
            st.session_state.histories[active_key] = []
            st.rerun()

    st.divider()

    # Suggestion chips (only when no history)
    if not history:
        st.markdown("**Try asking:**")
        chips = st.columns(4)
        suggestions = [
            f"What does {company.name} do?",
            "Summarise the financials",
            "What is the growth strategy?",
            "Latest news about this company",
        ]
        for i, sug in enumerate(suggestions):
            if chips[i].button(sug, use_container_width=True, key=f"chip_{i}"):
                st.session_state.pending_q = sug
                st.rerun()

    # Render chat history
    for msg in history:
        role = "assistant" if msg["role"] == "ai" else msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if msg.get("doc_chunks"):
                with st.expander(f"📎 {len(msg['doc_chunks'])} document chunks"):
                    for ch in msg["doc_chunks"]:
                        st.caption(f"[{ch['doc']}] dist={ch['dist']:.3f}")
                        st.text(textwrap.fill(ch["chunk"], 80))
            if msg.get("web_results"):
                with st.expander(f"🌐 {len(msg['web_results'])} web results"):
                    for r in msg["web_results"]:
                        st.markdown(f"**[{r['title']}]({r['url']})**")
                        st.caption(r["content"][:200] + "…" if len(r["content"]) > 200 else r["content"])

    # Chat input
    pending  = st.session_state.pop("pending_q", None)
    question = st.chat_input(f"Ask about {company.name}…") or pending

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        history.append({"role": "user", "content": question})

        embedder = load_embedder()

        # ① FAISS retrieval
        hits = store.search(question, embedder, k=TOP_K)
        ctx  = "\n\n".join(
            f"[{m['doc']} | {m['type']} | {m['date']} | dist={d:.3f}]\n{chunk}"
            for chunk, m, d in hits
        ) or "No document chunks retrieved."

        # ② Tavily web search (optional)
        web_results = []
        web_ctx     = ""
        if web_on and st.session_state.tavily_key:
            with st.spinner("🌐 Searching the web…"):
                search_q    = f"{company.name} {question}"
                web_results = tavily_search(search_q, st.session_state.tavily_key)
                web_ctx     = "\n\n".join(
                    f"[{r['title']}] ({r['url']})\n{r['content']}"
                    for r in web_results
                )

        # ③ Build system prompt
        if web_on and web_ctx:
            system = SYSTEM_WITH_WEB.format(
                profile=company.structured_profile(), context=ctx, web=web_ctx
            )
        else:
            system = SYSTEM_DOCS_ONLY.format(
                profile=company.structured_profile(), context=ctx
            )

        messages = [{"role": "system", "content": system}] + [
            {"role": "assistant" if h["role"] == "ai" else h["role"], "content": h["content"]}
            for h in history
        ]

        # ④ Stream from Groq
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            try:
                groq_client = Groq(api_key=st.session_state.groq_key)
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL, messages=messages,
                    max_tokens=1024, stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full += delta
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)

                # Show sources
                doc_chunks = [{"doc": m["doc"], "dist": d, "chunk": ch}
                              for ch, m, d in hits]
                if doc_chunks:
                    with st.expander(f"📎 {len(doc_chunks)} document chunks"):
                        for s in doc_chunks:
                            st.caption(f"[{s['doc']}] dist={s['dist']:.3f}")
                            st.text(textwrap.fill(s["chunk"], 80))
                if web_results:
                    with st.expander(f"🌐 {len(web_results)} web results"):
                        for r in web_results:
                            st.markdown(f"**[{r['title']}]({r['url']})**")
                            st.caption(r["content"][:200] + "…" if len(r["content"]) > 200 else r["content"])

                history.append({
                    "role": "ai", "content": full,
                    "doc_chunks": doc_chunks, "web_results": web_results,
                })

            except Exception as e:
                err = f"❌ Error: {e}"
                placeholder.error(err)
                history.append({"role": "ai", "content": err})

        st.session_state.histories[active_key] = history
