import streamlit as st
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ResearchVault",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --rv-bg: #070914;
    --rv-panel: rgba(14, 18, 34, 0.72);
    --rv-panel-strong: rgba(17, 21, 38, 0.90);
    --rv-border: rgba(167, 139, 250, 0.18);
    --rv-purple: #8B5CF6;
    --rv-indigo: #6366F1;
    --rv-violet: #A78BFA;
    --rv-teal: #5EEAD4;
    --rv-text: #F8FAFC;
    --rv-muted: #8F9AAF;
}

/* ============================================================
   LAYERED BACKGROUND
   ============================================================ */

.stApp {
    position: relative;
    overflow-x: hidden;
    min-height: 100vh;
    background:
        radial-gradient(circle at 50% -12%, rgba(139,92,246,0.20), transparent 31%),
        radial-gradient(circle at 7% 30%, rgba(99,102,241,0.10), transparent 27%),
        radial-gradient(circle at 92% 72%, rgba(45,212,191,0.055), transparent 25%),
        linear-gradient(135deg, #060811 0%, #090D1B 48%, #0B0F1D 100%);
    color: var(--rv-text);
    font-family: "DM Sans", Inter, ui-sans-serif, system-ui, sans-serif;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.25;
    background-image:
        linear-gradient(rgba(148,163,184,0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,0.055) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, black 0%, rgba(0,0,0,0.35) 62%, transparent 100%);
}

.stApp::after {
    content: "";
    position: fixed;
    width: 58vw;
    height: 58vw;
    left: 18%;
    top: -34%;
    pointer-events: none;
    z-index: 0;
    border-radius: 50%;
    background: radial-gradient(
        circle at 35% 45%,
        rgba(139,92,246,0.13),
        rgba(99,102,241,0.05) 34%,
        transparent 68%
    );
    filter: blur(26px);
    animation: rvAurora 18s ease-in-out infinite alternate;
}

@keyframes rvAurora {
    0%   { transform: translate3d(-3%,0,0) scale(.95); opacity:.55; }
    50%  { transform: translate3d(4%,3%,0) scale(1.05); opacity:.9; }
    100% { transform: translate3d(-1%,6%,0) scale(1); opacity:.62; }
}

@keyframes rvFloat {
    0%,100% { transform: translateY(0) rotate(-2deg); }
    50% { transform: translateY(-7px) rotate(2deg); }
}

@keyframes rvPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(139,92,246,0); opacity:.8; }
    50% { box-shadow: 0 0 0 7px rgba(139,92,246,.045); opacity:1; }
}

@keyframes rvShimmer {
    0% { transform: translateX(-120%); }
    100% { transform: translateX(120%); }
}

@keyframes rvRise {
    from { opacity:0; transform:translateY(10px); }
    to { opacity:1; transform:translateY(0); }
}

.main > div,
.block-container {
    position: relative;
    z-index: 1;
}

/* ============================================================
   LAYOUT / TYPOGRAPHY
   ============================================================ */

.block-container {
    max-width: 1160px;
    padding-top: 22px;
    padding-bottom: 130px;
}

#MainMenu, header, footer {
    visibility: hidden;
}

h1,h2,h3,h4,h5,h6 {
    color: #F8FAFC !important;
    font-family: "Space Grotesk","DM Sans",Inter,sans-serif;
    letter-spacing: -0.045em;
}

.stMarkdown p,
.stMarkdown li {
    color: #E5E7EB !important;
}

.stCaption {
    color: #9CA3AF !important;
}

/* ============================================================
   PRODUCT NAV
   ============================================================ */

.rv-nav {
    display:flex;
    align-items:center;
    justify-content:space-between;
    min-height:58px;
    padding:10px 14px 10px 16px;
    margin-bottom:28px;
    border:1px solid rgba(148,163,184,.12);
    border-radius:17px;
    background:rgba(9,12,25,.58);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    box-shadow:0 14px 45px rgba(0,0,0,.22);
}

.rv-brand {
    display:flex;
    align-items:center;
    gap:10px;
}

.rv-logo {
    width:31px;
    height:31px;
    display:grid;
    place-items:center;
    border-radius:10px;
    color:#fff;
    font-size:15px;
    background:linear-gradient(135deg,#4F46E5,#8B5CF6 65%,#5EEAD4);
    box-shadow:0 7px 25px rgba(99,102,241,.25);
    animation:rvPulse 4s ease-in-out infinite;
}

.rv-brand-name {
    color:#fff;
    font-family:"Space Grotesk",sans-serif;
    font-size:1.02rem;
    font-weight:700;
    letter-spacing:-.035em;
}

.rv-brand-caption {
    color:#737E95;
    font-size:.64rem;
    margin-top:-2px;
}

.rv-nav-right {
    display:flex;
    align-items:center;
    gap:8px;
}

.rv-nav-chip {
    color:#B9C1D2;
    background:rgba(255,255,255,.035);
    border:1px solid rgba(148,163,184,.12);
    border-radius:999px;
    padding:7px 10px;
    font-size:.68rem;
    font-weight:600;
}

.rv-live-dot {
    display:inline-block;
    width:6px;
    height:6px;
    margin-right:6px;
    border-radius:50%;
    background:#5EEAD4;
    box-shadow:0 0 12px rgba(94,234,212,.7);
}

/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    min-height:42px !important;
    background:rgba(15,23,42,.64) !important;
    color:#E5E7EB !important;
    border:1px solid rgba(148,163,184,.17) !important;
    border-radius:12px !important;
    font-weight:600 !important;
    box-shadow:0 5px 20px rgba(0,0,0,.16);
    transition:transform .18s ease,background .18s ease,border-color .18s ease,box-shadow .18s ease;
}

.stButton > button:hover {
    background:rgba(35,39,67,.78) !important;
    color:#fff !important;
    border-color:rgba(167,139,250,.42) !important;
    transform:translateY(-2px) scale(1.01);
    box-shadow:0 10px 30px rgba(0,0,0,.27),0 0 22px rgba(139,92,246,.07);
}

.stButton > button:active {
    transform:translateY(0) scale(.97);
}

.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#5B4AEF,#7C3AED 62%,#8B5CF6) !important;
    color:#fff !important;
    border:1px solid rgba(196,181,253,.16) !important;
    box-shadow:0 10px 32px rgba(99,102,241,.28);
}

.stButton > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#6D5CF5,#8B5CF6 62%,#A78BFA) !important;
    box-shadow:0 13px 38px rgba(99,102,241,.38);
}

/* ============================================================
   HERO
   ============================================================ */

.upload-space {
    height:58px;
}

.upload-kicker {
    text-align:center;
    color:#A78BFA !important;
    font-size:.67rem;
    font-weight:800;
    letter-spacing:.17em;
    text-transform:uppercase;
    animation:rvRise .6s ease both;
}

.upload-title {
    text-align:center;
    color:#fff !important;
    font-family:"Space Grotesk",sans-serif;
    font-size:clamp(3.25rem,6vw,5rem);
    font-weight:700;
    letter-spacing:-.075em;
    line-height:.98;
    margin-top:11px;
    animation:rvRise .7s .05s ease both;
}

.upload-gradient {
    background:linear-gradient(100deg,#fff 0%,#DDD6FE 34%,#A78BFA 65%,#5EEAD4 100%);
    background-size:180% auto;
    -webkit-background-clip:text;
    background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:rvTextShift 7s ease-in-out infinite alternate;
}

@keyframes rvTextShift {
    from { background-position:0% center; }
    to { background-position:100% center; }
}

.upload-description {
    max-width:650px;
    margin:18px auto 34px auto;
    text-align:center;
    color:#9CA3AF !important;
    font-size:.98rem;
    line-height:1.72;
    animation:rvRise .75s .1s ease both;
}

.hero-orbit {
    width:44px;
    height:44px;
    margin:-5px auto 18px auto;
    display:grid;
    place-items:center;
    border:1px solid rgba(167,139,250,.24);
    border-radius:14px;
    background:rgba(139,92,246,.075);
    color:#C4B5FD;
    box-shadow:0 0 35px rgba(139,92,246,.10);
    animation:rvFloat 4.5s ease-in-out infinite;
}

/* ============================================================
   UPLOAD CARD
   ============================================================ */

[data-testid="stFileUploader"] {
    position:relative;
    background:linear-gradient(145deg,rgba(18,22,40,.86),rgba(10,14,28,.78)) !important;
    border:1px solid rgba(167,139,250,.24) !important;
    border-radius:23px !important;
    padding:15px !important;
    box-shadow:0 28px 90px rgba(0,0,0,.42),0 0 60px rgba(99,102,241,.055);
    backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    animation:rvRise .8s .15s ease both;
}

[data-testid="stFileUploader"]:hover {
    border-color:rgba(167,139,250,.42) !important;
    box-shadow:0 30px 100px rgba(0,0,0,.48),0 0 75px rgba(139,92,246,.085);
}

[data-testid="stFileUploaderDropzone"] {
    position:relative;
    overflow:hidden;
    background:
        radial-gradient(circle at 50% 15%,rgba(139,92,246,.10),transparent 45%),
        linear-gradient(145deg,rgba(17,24,39,.90),rgba(13,18,34,.96)) !important;
    border:1.5px dashed rgba(167,139,250,.38) !important;
    border-radius:17px !important;
    min-height:205px;
    transition:border-color .25s ease,background .25s ease,transform .25s ease;
}

[data-testid="stFileUploaderDropzone"]::after {
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(110deg,transparent 35%,rgba(167,139,250,.065),transparent 65%);
    transform:translateX(-120%);
    pointer-events:none;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color:rgba(196,181,253,.70) !important;
    background:
        radial-gradient(circle at 50% 15%,rgba(139,92,246,.16),transparent 48%),
        linear-gradient(145deg,rgba(20,27,48,.94),rgba(13,18,34,.98)) !important;
    transform:translateY(-1px);
}

[data-testid="stFileUploaderDropzone"]:hover::after {
    animation:rvShimmer 1.25s ease;
}

[data-testid="stFileUploaderDropzone"] * {
    color:#D1D5DB !important;
}

[data-testid="stFileUploaderDropzone"] small {
    color:#8993A8 !important;
}

.rv-file-list {
    display:grid;
    gap:7px;
    margin-top:12px;
}

.rv-file-item {
    display:flex;
    align-items:center;
    gap:10px;
    padding:9px 11px;
    border-radius:11px;
    background:rgba(255,255,255,.035);
    border:1px solid rgba(148,163,184,.10);
    animation:rvRise .3s ease both;
}

.rv-pdf-badge {
    color:#FCA5A5;
    background:rgba(239,68,68,.10);
    border:1px solid rgba(248,113,113,.18);
    border-radius:7px;
    padding:3px 6px;
    font-size:.61rem;
    font-weight:800;
}

.rv-file-name {
    color:#DCE2ED;
    font-size:.77rem;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
}

/* ============================================================
   CARDS / STATUS
   ============================================================ */

[data-testid="stAlert"] {
    background:rgba(17,24,39,.72) !important;
    color:#E5E7EB !important;
    border:1px solid rgba(94,234,212,.13) !important;
    border-radius:13px !important;
}

[data-testid="stAlert"] p {
    color:#E5E7EB !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background:rgba(14,18,34,.66) !important;
    border:1px solid rgba(167,139,250,.15) !important;
    border-radius:16px !important;
    backdrop-filter:blur(12px);
    -webkit-backdrop-filter:blur(12px);
}

.paper-name {
    color:#F8FAFC !important;
    font-weight:650;
    font-size:.92rem;
}

.paper-label {
    color:#7C879D !important;
    font-size:.63rem;
    font-weight:800;
    letter-spacing:.12em;
    text-transform:uppercase;
}

.suggestion-title {
    text-align:center;
    color:#8E99AE;
    font-size:.62rem;
    font-weight:800;
    letter-spacing:.14em;
    text-transform:uppercase;
}

/* ============================================================
   CHAT / LOADING
   ============================================================ */

[data-testid="stChatMessage"] {
    background:transparent !important;
    border:none !important;
    padding-top:13px !important;
    padding-bottom:13px !important;
    animation:rvRise .35s ease both;
}

[data-testid="stChatMessage"] p {
    color:#F1F5F9 !important;
    line-height:1.78 !important;
}

[data-testid="stChatMessage"] li {
    color:#E5E7EB !important;
    line-height:1.72 !important;
}

[data-testid="stChatMessage"] strong {
    color:#fff !important;
}

[data-testid="stChatMessage"] code {
    background:#111827 !important;
    color:#C4B5FD !important;
    border-radius:5px;
    padding:2px 5px;
}

[data-testid="stChatMessageAvatarIcon"] {
    background:linear-gradient(135deg,#312E81,#6D28D9) !important;
    color:#fff !important;
}

[data-testid="stChatInput"] {
    background:rgba(12,18,35,.88) !important;
    border:none !important;
    outline:none !important;
    border-radius:18px !important;
    box-shadow:0 18px 60px rgba(0,0,0,.43),0 0 30px rgba(99,102,241,.05) !important;
    padding:3px !important;
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
}

[data-testid="stChatInput"]:focus-within {
    border:none !important;
    outline:none !important;
    box-shadow:0 18px 60px rgba(0,0,0,.43),0 0 30px rgba(99,102,241,.05) !important;
}

[data-testid="stChatInput"] textarea {
    background:transparent !important;
    border:none !important;
    outline:none !important;
    box-shadow:none !important;
    color:#fff !important;
    -webkit-text-fill-color:#fff !important;
    caret-color:#A78BFA !important;
    font-size:.94rem !important;
}

/* ============================================================
   CHAT INPUT — remove browser / Grammarly visual overlays
   ============================================================ */

[data-testid="stChatInput"],
[data-testid="stChatInput"]:focus,
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus,
[data-testid="stChatInput"] > div:focus-within {
    border: 0 !important;
    outline: 0 !important;
    box-shadow: 0 18px 60px rgba(0,0,0,.43), 0 0 30px rgba(99,102,241,.05) !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible,
[data-testid="stChatInput"] textarea:hover,
[data-testid="stChatInput"] textarea:active {
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
    -webkit-appearance: none !important;
    appearance: none !important;
}

/* Grammarly / writing-assistant injected elements */
grammarly-desktop-integration,
grammarly-extension,
.grammarly-extension,
div[class*="grammarly"],
[data-gramm_editor],
[data-enable-grammarly] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color:#7C879D !important;
    -webkit-text-fill-color:#7C879D !important;
    opacity:1 !important;
}

[data-testid="stChatInput"] button {
    background:linear-gradient(135deg,#5B4AEF,#7C3AED) !important;
    color:#fff !important;
    border:none !important;
    border-radius:11px !important;
    transition:transform .15s ease;
}

[data-testid="stChatInput"] button:hover { transform:scale(1.05); }
[data-testid="stChatInput"] button:active { transform:scale(.95); }

[data-testid="stSpinner"] { color:#C4B5FD !important; }
[data-testid="stSpinner"] svg { stroke:#A78BFA !important; }

/* ============================================================
   EVIDENCE
   ============================================================ */

[data-testid="stExpander"] {
    background:#0D1426 !important;
    border:1px solid rgba(167,139,250,.19) !important;
    border-radius:14px !important;
    margin-bottom:10px !important;
    overflow:hidden !important;
    box-shadow:0 7px 28px rgba(0,0,0,.20);
    transition:border-color .2s ease,transform .2s ease;
}

[data-testid="stExpander"]:hover {
    border-color:rgba(167,139,250,.33) !important;
}

[data-testid="stExpander"] summary {
    background:#10182D !important;
    color:#F8FAFC !important;
    border:none !important;
    font-size:.82rem !important;
    font-weight:650 !important;
}

[data-testid="stExpander"] summary:hover {
    background:#151F39 !important;
    color:#fff !important;
}

[data-testid="stExpander"] > details,
[data-testid="stExpanderDetails"] {
    background:#0D1426 !important;
    color:#E5E7EB !important;
}

[data-testid="stExpanderDetails"] {
    border-top:1px solid rgba(167,139,250,.12) !important;
}

[data-testid="stExpander"] p {
    color:#DDE3EF !important;
    line-height:1.75 !important;
}

[data-testid="stExpander"] blockquote {
    background:#111B32 !important;
    border-left:3px solid #8B5CF6 !important;
    color:#DDE3EF !important;
    border-radius:0 9px 9px 0;
    padding:14px 16px;
    margin:10px 0;
}

[data-testid="stExpander"] blockquote p {
    color:#DDE3EF !important;
}

/* ============================================================
   FOOTER / SCROLLBAR / MOBILE
   ============================================================ */

.rv-footer {
    margin-top:65px;
    padding-top:18px;
    border-top:1px solid rgba(148,163,184,.10);
    display:flex;
    justify-content:space-between;
    color:#59647A;
    font-size:.66rem;
    letter-spacing:.03em;
}

.rv-footer strong { color:#8E99AE; }

::-webkit-scrollbar { width:8px; }
::-webkit-scrollbar-track { background:#070B16; }
::-webkit-scrollbar-thumb { background:#252E46; border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background:#3B4665; }

@media (max-width:700px) {
    .block-container { padding-top:12px; padding-bottom:105px; }
    .upload-space { height:28px; }
    .upload-title { font-size:2.65rem; }
    .upload-description { font-size:.9rem; margin-bottom:26px; }
    .rv-nav { margin-bottom:20px; }
    .rv-nav-chip { display:none; }
    .hero-orbit { margin-bottom:13px; }
    .rv-footer { flex-direction:column; gap:6px; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <script>
    (() => {
        const cleanResearchVaultInput = () => {
            const doc = window.parent.document;

            // Disable browser writing-assistant hooks on the actual chat textarea.
            doc.querySelectorAll('[data-testid="stChatInput"] textarea').forEach(t => {
                t.setAttribute('data-gramm', 'false');
                t.setAttribute('data-gramm_editor', 'false');
                t.setAttribute('data-enable-grammarly', 'false');
                t.setAttribute('spellcheck', 'false');
                t.setAttribute('autocomplete', 'off');
                t.setAttribute('autocorrect', 'off');
                t.setAttribute('autocapitalize', 'off');

                t.style.setProperty('border', '0', 'important');
                t.style.setProperty('outline', '0', 'important');
                t.style.setProperty('box-shadow', 'none', 'important');
            });

            // Remove Grammarly / writing-assistant DOM overlays if injected.
            doc.querySelectorAll(
                'grammarly-desktop-integration, grammarly-extension, ' +
                '.grammarly-extension, div[class*="grammarly"]'
            ).forEach(el => el.remove());
        };

        cleanResearchVaultInput();

        if (!window.__researchVaultGrammarlyObserver) {
            window.__researchVaultGrammarlyObserver =
                new MutationObserver(cleanResearchVaultInput);

            window.__researchVaultGrammarlyObserver.observe(
                window.parent.document.body,
                { childList: true, subtree: true }
            );
        }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "paper_names" not in st.session_state:
    st.session_state.paper_names = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ============================================================
# BACKEND HELPERS
# ============================================================

def backend_available():

    try:

        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


def delete_backend_session():

    session_id = st.session_state.session_id

    if not session_id:
        return

    try:

        requests.delete(
            f"{BACKEND_URL}/session/{session_id}",
            timeout=10,
        )

    except requests.RequestException:

        pass


def reset_session():

    delete_backend_session()

    st.session_state.session_id = None
    st.session_state.paper_names = []
    st.session_state.messages = []

    st.session_state.uploader_key += 1

    st.rerun()


# ============================================================
# UPLOAD
# ============================================================

def upload_papers(uploaded_files):
    files = [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            ),
        )
        for uploaded_file in uploaded_files
    ]

    try:
        response = requests.post(
            f"{BACKEND_URL}/session/upload",
            files=files,
            timeout=180,
        )

    except requests.RequestException:
        st.error(
            "ResearchVault could not connect to the backend."
        )
        return

    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail",
                "The papers could not be processed.",
            )
        except Exception:
            detail = "The papers could not be processed."

        st.error(detail)
        return

    data = response.json()

    st.session_state.session_id = data["session_id"]
    st.session_state.paper_names = data.get("paper_names", [])
    st.session_state.messages = []

    st.toast(
        "Research session ready ✦",
        icon="✨",
    )

    st.rerun()


# ============================================================
# ASK QUESTION
# ============================================================

def submit_question(question):

    question = question.strip()

    if not question:
        return

    if not st.session_state.session_id:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:

        with st.spinner(
            "ResearchVault is thinking through the evidence..."
        ):

            response = requests.post(
                f"{BACKEND_URL}/session/ask",
                json={
                    "session_id": st.session_state.session_id,
                    "question": question,
                },
                timeout=180,
            )

    except requests.RequestException:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I couldn't connect to the ResearchVault backend. "
                    "Please make sure FastAPI is running."
                ),
                "sources": [],
            }
        )

        st.rerun()

        return

    if response.status_code == 404:

        st.session_state.session_id = None
        st.session_state.paper_names = []
        st.session_state.messages = []

        st.warning(
            "This research session has expired. "
            "Please upload the paper again."
        )

        st.rerun()

        return

    if response.status_code != 200:

        try:

            detail = response.json().get(
                "detail",
                "Something went wrong while answering.",
            )

        except Exception:

            detail = (
                "Something went wrong while answering."
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": detail,
                "sources": [],
            }
        )

        st.rerun()

        return

    data = response.json()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data.get(
                "answer",
                "No answer was returned.",
            ),
            "sources": data.get(
                "sources",
                [],
            ),
        }
    )

    st.rerun()


# ============================================================
# PRODUCT CHROME
# ============================================================

def render_navbar(active_session=False):
    session_label = (
        "Active research session"
        if active_session
        else "Private research workspace"
    )

    st.markdown(
        f"""
        <div class="rv-nav">
            <div class="rv-brand">
                <div class="rv-logo">✦</div>
                <div>
                    <div class="rv-brand-name">ResearchVault</div>
                    <div class="rv-brand-caption">Multi-paper research intelligence</div>
                </div>
            </div>
            <div class="rv-nav-right">
                <div class="rv-nav-chip">
                    <span class="rv-live-dot"></span>{session_label}
                </div>
                <div class="rv-nav-chip">v1.0</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="rv-footer">
            <span><strong>ResearchVault</strong> · Research Workspace</span>
            <span>Temporary sessions · grounded evidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD SCREEN
# ============================================================

def render_upload_screen():

    render_navbar(active_session=False)

    st.markdown(
        '<div class="upload-space"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="upload-kicker">RESEARCH WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 class="upload-title">
            Turn papers into
            <span class="upload-gradient">answers.</span>
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-orbit">✧</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-description">
            Your research papers, transformed into instant insights and evidence-backed answers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 4, 1])

    with center:

        uploaded_files = st.file_uploader(
            "Upload your research papers",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if uploaded_files:

            paper_count = len(uploaded_files)

            st.info(
                f"📚 **{paper_count} paper"
                f"{'' if paper_count == 1 else 's'}** ready to research."
            )

            file_markup = '<div class="rv-file-list">'

            for uploaded_file in uploaded_files:
                safe_name = (
                    uploaded_file.name
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                file_markup += (
                    '<div class="rv-file-item">'
                    '<span class="rv-pdf-badge">PDF</span>'
                    f'<span class="rv-file-name">{safe_name}</span>'
                    '</div>'
                )

            file_markup += '</div>'

            st.markdown(
                file_markup,
                unsafe_allow_html=True,
            )

            if st.button(
                "Start research session  →",
                type="primary",
                use_container_width=True,
            ):

                if not backend_available():

                    st.error(
                        "FastAPI is not running. Start the backend first."
                    )

                else:

                    with st.spinner(
                        "Preparing your research workspace..."
                    ):
                        upload_papers(uploaded_files)

        else:

            st.caption(
                "PDF files · grounded answers · page-level evidence"
            )

    render_footer()


# ============================================================
# PAPER HEADER
# ============================================================

def render_paper_header():

    render_navbar(active_session=True)

    with st.container(border=True):

        paper_icon, paper_info, action_col = st.columns(
            [0.55, 5.6, 1.2],
            vertical_alignment="center",
        )

        with paper_icon:
            st.markdown("📚")

        with paper_info:

            st.markdown(
                '<div class="paper-label">RESEARCH SESSION</div>',
                unsafe_allow_html=True,
            )

            paper_names = st.session_state.paper_names

            if len(paper_names) == 1:
                paper_text = paper_names[0]
            else:
                paper_text = "  ·  ".join(paper_names)

            st.markdown(
                f'<div class="paper-name">{paper_text}</div>',
                unsafe_allow_html=True,
            )

        with action_col:

            if st.button(
                "End session",
                use_container_width=True,
            ):
                reset_session()

    st.write("")


# ============================================================
# EMPTY WORKSPACE
# ============================================================

def render_empty_workspace():

    st.write("")

    st.markdown(
        '<div style="text-align:center;">'
        '<div style="font-size:1.7rem;">✦</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h2 style="
            text-align:center;
            font-size:2.2rem;
            margin-bottom:8px;
        ">
            What do you want to discover?
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="
            text-align:center;
            color:#85858E;
            font-size:0.92rem;
            margin-bottom:38px;
        ">
            Ask about the papers' ideas, methods, results,
            contributions, comparisons, or limitations.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="suggestion-title">START EXPLORING</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    suggestions = [
    (
        c1,
        "✦  What matters?",
        "What are the most important ideas I should take away from these papers?",
    ),
    (
        c2,
        "⌁  Compare approaches",
        "How do the approaches in these papers differ?",
    ),
    (
        c3,
        "◈  Key findings",
        "What are the key findings and conclusions of these papers?",
    ),
    (
        c4,
        "△  Research gaps",
        "What limitations or open research gaps do these papers identify?",
    ),
]

    for column, label, question in suggestions:

        with column:

            if st.button(
                label,
                use_container_width=True,
            ):

                submit_question(
                    question
                )


# ============================================================
# EVIDENCE
# ============================================================

def render_sources(sources):

    if not sources:
        return

    st.markdown(
        "---"
    )

    st.caption(
        "SUPPORTING EVIDENCE"
    )

    for index, source in enumerate(
        sources,
        start=1,
    ):

        paper = source.get(
            "paper_name",
            "Research paper",
        )

        page = source.get(
            "page_number",
            "—",
        )

        passage = source.get(
            "passage",
            "",
        )

        with st.expander(
            f"Evidence {index}   ·   {paper}   ·   Page {page}"
        ):

            st.caption(
                f"{paper}  ·  Page {page}"
            )

            st.markdown(
                f"> {passage}"
            )


# ============================================================
# CHAT HISTORY
# ============================================================

def render_messages():

    for message in st.session_state.messages:

        if message["role"] == "user":

            with st.chat_message(
                "user",
                avatar="👤",
            ):

                st.markdown(
                    message["content"]
                )

        else:

            with st.chat_message(
                "assistant",
                avatar="✨",
            ):

                st.caption(
                    "RESEARCHVAULT"
                )

                st.markdown(
                    message["content"]
                )

                render_sources(
                    message.get(
                        "sources",
                        [],
                    )
                )


# ============================================================
# RESEARCH WORKSPACE
# ============================================================

def render_research_workspace():

    render_paper_header()

    if not st.session_state.messages:
        render_empty_workspace()
    else:
        render_messages()

    question = st.chat_input(
        "Ask, compare, or explore your research"
    )

    if question:
        submit_question(question)

    render_footer()


# ============================================================
# MAIN
# ============================================================

def main():

    if st.session_state.session_id:

        render_research_workspace()

    else:

        render_upload_screen()


if __name__ == "__main__":
    main()