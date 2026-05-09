"""
🕵️ Fake News Detection — Streamlit Web App
Based on: fake_news_detection_Final.ipynb
Models: Logistic Regression, LinearSVC, LightGBM, Naive Bayes, Hybrid LR (TF-IDF + SBERT)
Languages: Bangla (bn) + English (en)
"""

import streamlit as st
import os, re, warnings, time
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import requests

warnings.filterwarnings("ignore")

# ─── Google Drive download helper ───────────────────────────────────────────
DRIVE_FILES = {
    "train_balanced_clean.csv": "1ItZ9lIq8Lo3vkn754_Q-LDJb-cjWrivV",
    "test_clean.csv":           "12MpwM_VI5f2HBYpLUHoCQliusvH8LaDn",
}

def download_from_drive(file_id: str, dest_path: str):
    """Download a public Google Drive file using gdown (handles large files)."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False, fuzzy=True)

    # Sanity check
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000:
        raise RuntimeError(
            f"Failed to download {dest_path}. "
            "Make sure Google Drive sharing is set to 'Anyone with the link'."
        )

def ensure_csv_files():
    """Download CSVs from Drive if not already present."""
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            with st.spinner(f"⬇️ Downloading {filename} from Google Drive…"):
                download_from_drive(file_id, filename)
    return True

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🕵️ Fake News Detector",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark editorial theme */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #f5f5f5 0%, #9ca3c8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.main-header .subtitle {
    color: #6b7280;
    font-size: 0.95rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}

/* Verdict cards */
.verdict-real {
    background: linear-gradient(135deg, #052e16 0%, #064e1c 100%);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 40px rgba(34,197,94,0.15);
}
.verdict-fake {
    background: linear-gradient(135deg, #2d0a0a 0%, #4a1010 100%);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 40px rgba(239,68,68,0.15);
}
.verdict-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}
.verdict-real .verdict-label { color: #4ade80; }
.verdict-fake .verdict-label { color: #f87171; }

.confidence-bar-wrap {
    margin-top: 1.2rem;
}
.conf-label {
    font-size: 0.8rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.4rem;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.conf-name {
    width: 48px;
    font-size: 0.8rem;
    color: #d1d5db;
    text-align: right;
}
.conf-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    overflow: hidden;
}
.conf-fill-fake {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #f97316);
    border-radius: 4px;
    transition: width 0.6s ease;
}
.conf-fill-real {
    height: 100%;
    background: linear-gradient(90deg, #22c55e, #10b981);
    border-radius: 4px;
    transition: width 0.6s ease;
}
.conf-pct {
    width: 44px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #e5e7eb;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: #181b23;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-card .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #c7d2fe;
}
.metric-card .metric-lbl {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b7280;
    margin-top: 2px;
}

/* Input area overrides */
.stTextArea textarea {
    background: #181b23 !important;
    border: 1.5px solid #2a2d3a !important;
    color: #e8eaf0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

.stSelectbox > div > div {
    background: #181b23 !important;
    border: 1.5px solid #2a2d3a !important;
    color: #e8eaf0 !important;
    border-radius: 10px !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0f14 !important;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] .sidebar-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4b5563;
    margin: 1.5rem 0 0.5rem 0;
}

/* History item */
.history-item {
    background: #181b23;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 8px;
    font-size: 0.82rem;
}
.history-item .hi-text {
    color: #9ca3af;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 220px;
}
.history-item .hi-verdict {
    font-weight: 600;
    font-size: 0.75rem;
}
.hi-real { color: #4ade80; }
.hi-fake { color: #f87171; }

/* Loading spinner text */
.loading-msg {
    text-align: center;
    color: #6b7280;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 1rem 0;
}

.divider {
    border: none;
    border-top: 1px solid #1e2130;
    margin: 1.5rem 0;
}

.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 0 4px;
}
.tag-bn { background: #1e3a5f; color: #93c5fd; }
.tag-en { background: #1a2e1a; color: #86efac; }
</style>
""", unsafe_allow_html=True)


# ─── NLTK & model loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nltk():
    import nltk
    for pkg in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
        nltk.download(pkg, quiet=True)
    return True

@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """Train all models exactly as in the notebook — cached after first run."""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    from sentence_transformers import SentenceTransformer

    RANDOM_SEED = 42
    TEXT_COL = 'combined_text'
    LABEL_COL = 'label'
    LANG_COL  = 'language'
    LABEL_MAP = {'positive': 1, 'negative': 0}

    # ── Data loading ──
    train_path = 'train_balanced_clean.csv'
    test_path  = 'test_clean.csv'

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return None, None, None, None, "CSV files not found"

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    for df in [df_train, df_test]:
        df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
        df[LABEL_COL] = df[LABEL_COL].map(LABEL_MAP)
        df.reset_index(drop=True, inplace=True)

    # ── Preprocessing ──
    en_stop = set(stopwords.words('english'))
    bn_stop = {
        'এই','সেই','তার','এবং','বা','কিন্তু','যে','এটি','এটা','হয়','হচ্ছে',
        'আছে','ছিল','করা','করে','করেছে','থেকে','দিয়ে','জন্য','হবে','তাই',
        'আর','ও','না','নয়','কি','কে','কার','যখন','তখন','যেন','মতো','মধ্যে',
        'সব','সে','আমি','তুমি','আমরা','তারা','আপনি','একটি','একটা','একজন'
    }
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text: str, lang: str = 'en') -> str:
        text = str(text).lower().strip()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        if lang == 'en':
            text = re.sub(r'[^a-z\s]', ' ', text)
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(t) for t in tokens
                      if t not in en_stop and len(t) > 2]
        else:
            text = re.sub(r'[^\u0980-\u09FF\s]', ' ', text)
            tokens = text.split()
            tokens = [t for t in tokens if t not in bn_stop and len(t) > 1]
        return ' '.join(tokens)

    df_train['processed_text'] = df_train.apply(
        lambda r: preprocess_text(r[TEXT_COL], r.get(LANG_COL, 'en')), axis=1)
    df_test['processed_text'] = df_test.apply(
        lambda r: preprocess_text(r[TEXT_COL], r.get(LANG_COL, 'en')), axis=1)

    # ── TF-IDF ──
    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                            min_df=2, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(df_train['processed_text'])
    y_train = df_train[LABEL_COL].values

    # ── SBERT embeddings ──
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    train_texts = df_train['processed_text'].tolist()
    X_train_sbert = sbert_model.encode(
        train_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    X_train_hybrid = hstack([X_train_tfidf, csr_matrix(X_train_sbert)])

    # ── Train all models ──
    models_cfg = {
        "Hybrid LR (TF-IDF + SBERT)": (
            LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
            X_train_hybrid),
        "Logistic Regression": (
            LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
            X_train_tfidf),
        "LinearSVC": (
            LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED),
            X_train_tfidf),
        "LightGBM": (
            lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1,
                               num_leaves=63, random_state=RANDOM_SEED,
                               verbosity=-1),
            X_train_tfidf),
        "Naive Bayes": (
            MultinomialNB(alpha=0.1),
            X_train_tfidf),
    }

    trained_models = {}
    for name, (clf, X) in models_cfg.items():
        clf.fit(X, y_train)
        trained_models[name] = clf

    return trained_models, tfidf, sbert_model, preprocess_text, None


# ─── Session state ───────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🕵️ Fake News Detector</h1>
    <div class="subtitle">Bangla &amp; English · TF-IDF + SBERT Hybrid</div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar: settings + history ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        ["Hybrid LR (TF-IDF + SBERT)",
         "Logistic Regression",
         "LinearSVC",
         "LightGBM",
         "Naive Bayes"],
        index=0,
    )

    lang_choice = st.selectbox(
        "Language / ভাষা",
        [("English", "en"), ("Bangla (বাংলা)", "bn")],
        format_func=lambda x: x[0],
    )
    lang_code = lang_choice[1]

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#4b5563; line-height:1.6;'>
    <b style='color:#6b7280;'>Model notes</b><br>
    🏆 <b>Hybrid LR</b> — Best overall (TF-IDF + SBERT)<br>
    ⚡ <b>LinearSVC</b> — Fastest, great accuracy<br>
    🌳 <b>LightGBM</b> — Strong on structured text<br>
    📊 <b>Logistic Reg</b> — Interpretable baseline<br>
    🔤 <b>Naive Bayes</b> — Lightweight keyword model
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Recent Checks")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-8:]):
            v_class = "hi-real" if item['verdict'] == 'REAL' else "hi-fake"
            v_icon  = "🟢" if item['verdict'] == 'REAL' else "🔴"
            st.markdown(f"""
            <div class="history-item">
                <div class="hi-text">{item['text'][:55]}…</div>
                <div class="hi-verdict {v_class}">{v_icon} {item['verdict']} · {item['model'][:12]}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#4b5563; font-size:0.8rem;'>No checks yet.</span>",
                    unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("🗑 Clear history"):
            st.session_state.history = []
            st.rerun()


# ─── Auto-download CSV files from Google Drive ───────────────────────────────
ensure_csv_files()

# ─── Main prediction UI ───────────────────────────────────────────────────────
if True:
    # Load / train models
    with st.spinner("⚙️ Loading models (first run may take ~2 min to train)…"):
        load_nltk()
        trained_models, tfidf, sbert_model, preprocess_fn, err = load_models_and_data()

    if err:
        st.error(f"❌ {err}")
        st.stop()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Text input ──
    lang_badge = f'<span class="tag tag-{"bn" if lang_code=="bn" else "en"}">{lang_code.upper()}</span>'
    st.markdown(f"### 📰 Enter News Text {lang_badge}", unsafe_allow_html=True)

    placeholder = (
        "এখানে বাংলা সংবাদ শিরোনাম বা প্রতিবেদন লিখুন…"
        if lang_code == 'bn' else
        "Enter a news headline or article here…"
    )

    news_text = st.text_area(
        label="news_input",
        label_visibility="collapsed",
        placeholder=placeholder,
        height=150,
        key="news_text_area",
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        check_btn = st.button("🔍  Check News", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state['news_text_area'] = ''
            st.rerun()

    # ── Prediction ──
    if check_btn:
        if not news_text.strip():
            st.warning("⚠️ Please enter some news text first.")
        else:
            from sklearn.linear_model import LogisticRegression

            with st.spinner("🔍 Analysing…"):
                # Preprocess
                proc = preprocess_fn(news_text, lang_code)

                if not proc.strip():
                    st.error("❌ Text is empty after preprocessing. Please enter a longer input.")
                    st.stop()

                # Features
                tfidf_feat = tfidf.transform([proc])
                sbert_feat = sbert_model.encode(
                    [proc], batch_size=1, convert_to_numpy=True)

                clf = trained_models[model_choice]

                if model_choice == "Hybrid LR (TF-IDF + SBERT)":
                    feat = hstack([tfidf_feat, csr_matrix(sbert_feat)])
                else:
                    feat = tfidf_feat

                pred = clf.predict(feat)[0]
                is_real = (pred == 1)

                # Confidence
                proba_fake, proba_real = None, None
                if hasattr(clf, 'predict_proba'):
                    proba = clf.predict_proba(feat)[0]
                    proba_fake, proba_real = proba[0] * 100, proba[1] * 100
                elif hasattr(clf, 'decision_function'):
                    raw = clf.decision_function(feat)[0]
                    p = 1 / (1 + np.exp(-raw))
                    proba_real = p * 100
                    proba_fake = (1 - p) * 100

            # ── Verdict display ──
            verdict_class = "verdict-real" if is_real else "verdict-fake"
            verdict_label = "✅ REAL NEWS" if is_real else "❌ FAKE NEWS"
            verdict_emoji = "🟢" if is_real else "🔴"

            conf_html = ""
            if proba_fake is not None:
                conf_html = f"""
                <div class="confidence-bar-wrap">
                    <div class="conf-label">Confidence</div>
                    <div class="conf-row">
                        <div class="conf-name">FAKE</div>
                        <div class="conf-track">
                            <div class="conf-fill-fake" style="width:{proba_fake:.1f}%"></div>
                        </div>
                        <div class="conf-pct">{proba_fake:.1f}%</div>
                    </div>
                    <div class="conf-row">
                        <div class="conf-name">REAL</div>
                        <div class="conf-track">
                            <div class="conf-fill-real" style="width:{proba_real:.1f}%"></div>
                        </div>
                        <div class="conf-pct">{proba_real:.1f}%</div>
                    </div>
                </div>
                """

            preview = news_text[:160] + ("…" if len(news_text) > 160 else "")

            st.markdown(f"""
            <div class="{verdict_class}">
                <div class="verdict-label">{verdict_emoji}  {verdict_label}</div>
                <div style="color:#9ca3af; font-size:0.82rem; margin-top:4px;">
                    Model: <b style="color:#c7d2fe;">{model_choice}</b> &nbsp;·&nbsp;
                    Language: <b style="color:#c7d2fe;">{lang_code.upper()}</b>
                </div>
                {conf_html}
                <div style="margin-top:1.2rem; padding-top:1rem;
                            border-top:1px solid rgba(255,255,255,0.08);
                            font-size:0.8rem; color:#6b7280; font-style:italic;">
                    "{preview}"
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                'text': news_text,
                'verdict': 'REAL' if is_real else 'FAKE',
                'model': model_choice,
                'lang': lang_code,
            })

    # ── Quick demo examples ──
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🧪 Quick Demo Examples")
    st.markdown("<div style='color:#6b7280; font-size:0.85rem; margin-bottom:1rem;'>Click any example to auto-fill the text box:</div>",
                unsafe_allow_html=True)

    examples = [
        ("🌍 EN — Likely Fake",
         "Scientists confirm that drinking water from plastic bottles causes instant cancer. Share before they delete this!",
         "en"),
        ("🌍 EN — Likely Real",
         "The International Monetary Fund released its World Economic Outlook report forecasting moderate global growth.",
         "en"),
        ("🇧🇩 BN — Likely Fake",
         "বাংলাদেশে বিনামূল্যে ৫জি ইন্টারনেট চালু হচ্ছে আগামীকাল থেকে, সবাইকে শেয়ার করতে বলা হয়েছে।",
         "bn"),
        ("🇧🇩 BN — Likely Real",
         "বাংলাদেশ ব্যাংক সুদের হার অপরিবর্তিত রেখেছে এবং মুদ্রাস্ফীতি নিয়ন্ত্রণে নতুন নির্দেশনা জারি করেছে।",
         "bn"),
    ]

    ex_cols = st.columns(2)
    for i, (label, text, lang) in enumerate(examples):
        with ex_cols[i % 2]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state['news_text_area'] = text
                st.rerun()

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#374151; font-size:0.75rem; padding-bottom:2rem;'>
    Fake News Detection · Bangla + English · paraphrase-multilingual-MiniLM-L12-v2
</div>
""", unsafe_allow_html=True)
