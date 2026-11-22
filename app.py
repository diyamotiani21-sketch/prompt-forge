import streamlit as st, pandas as pd
from techniques import TECHNIQUES, TASK_INSTRUCTIONS
from llm_client import call_llm
from metrics import extract_answer, approx_tokens
from ladder import VARIANTS, RUBRIC, build, judge


# ---------------- CUSTOM UI CONFIGURATION ----------------

st.set_page_config(
    page_title="PromptForge | AI Prompt Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>


/* CUSTOM TYPOGRAPHY - TIMES NEW ROMAN */

html, body, [class*="st-"],
.stMarkdown, .stTextInput, .stTextArea,
.stSelectbox, .stMultiSelect, .stButton,
.stTabs, label {
    font-family: "Times New Roman", Times, serif !important;
}

h1 {
    font-size: 50px !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 28px !important;
    font-weight: 700 !important;
}

h3 {
    font-size: 22px !important;
    font-weight: 600 !important;
}

p, label {
    font-size: 16px;
    line-height: 1.6;
}



/* Main application background */
.stApp {
    background-color: #0B1020;
    color: #F1F5F9;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #11182B;
    border-right: 1px solid #26314B;
}

/* Headings */
h1, h2, h3 {
    color: #F8FAFC !important;
}

/* Paragraphs and labels */
p, label, .stMarkdown {
    color: #CBD5E1;
}

/* Input fields */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background-color: #151F35;
    color: #F8FAFC;
    border: 1px solid #34415F;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    border: none;
    transform: translateY(-2px);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #CBD5E1;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #A78BFA !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #26314B;
    border-radius: 10px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #151F35;
    border: 1px solid #26314B;
    padding: 16px;
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #26314B;
}

/* -------- PROFESSIONAL INPUT FIELDS -------- */

.stTextInput input,
.stTextArea textarea {
    background-color: #F8FAFC !important;
    color: #172033 !important;
    border: 1.5px solid #94A3B8 !important;
    border-radius: 10px !important;
    font-family: "Times New Roman", serif !important;
    font-size: 19px !important;
    padding: 12px !important;
}

/* Placeholder visibility */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

/* Input focus */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border: 2px solid #8B5CF6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.15) !important;
}

/* -------- GENERATED PROMPT BOXES -------- */

[data-testid="stCode"] {
    background-color: #F1F5F9 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

[data-testid="stCode"] pre,
[data-testid="stCode"] code,
[data-testid="stCode"] span {
    color: #172033 !important;
    background-color: transparent !important;
    font-family: "Times New Roman", serif !important;
    font-size: 18px !important;
    line-height: 1.7 !important;
}

/* -------- RESULT TEXT -------- */

.stMarkdown p {
    font-size: 17px;
    line-height: 1.8;
}


</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------


# ---------------- DASHBOARD HEADER ----------------

st.markdown("""
<div style="
    padding: 25px 30px;
    background: linear-gradient(120deg, #172554, #312E81, #581C87);
    border: 1px solid #4338CA;
    border-radius: 18px;
    margin-bottom: 25px;
">
    <div style="font-size: 14px; color: #C4B5FD; letter-spacing: 2px;">
        AI PROMPT ENGINEERING WORKSPACE
    </div>
    <h1 style="font-size: 38px; margin: 10px 0; color: white;">
        ⚡ PromptForge
    </h1>
    <p style="font-size: 16px; color: #E0E7FF; margin-bottom: 0;">
        Experiment. Compare. Optimize.
        Discover how prompt wording impacts AI responses.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR BRANDING ----------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 15px 5px 25px 5px;">
        <h2 style="color:#A78BFA; margin-bottom:0;">
            ⚡ PromptForge
        </h2>
        <p style="color:#94A3B8; font-size:12px;">
            AI PROMPT ENGINEERING LAB
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Navigation")
    st.caption("Choose a workspace using the tabs.")
    st.markdown("""
    <div style="
        background:#1E293B;
        padding:12px;
        border-radius:10px;
        margin-top:10px;
        color:#C4B5FD;
    ">
        🧪 Prompt Ladder
    </div>
    <div style="
        background:#1E293B;
        padding:12px;
        border-radius:10px;
        margin-top:8px;
        color:#C4B5FD;
    ">
        ⚖️ Technique Comparator
    </div>
    <div style="
        background:#1E293B;
        padding:12px;
        border-radius:10px;
        margin-top:8px;
        color:#C4B5FD;
    ">
        📊 Benchmark Results
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Version 1.0 · PromptForge")

tab1, tab2, tab3 = st.tabs(["Prompt Ladder (one task, 5 prompts)", "Technique comparator", "Benchmark results"])

with tab1:
    st.header("🧪 Prompt Engineering Ladder")
    st.caption(
        "Explore how progressively improving a prompt "
        "can change the quality of AI responses."
    )

    topic = st.text_input("Enter your topic", "photosynthesis")
    aud = st.text_input("Target audience", "a Class 8 student")

    st.info("Five prompt variations will be generated and evaluated.")

    if st.button("Run the ladder"):
        if not topic.strip() or not aud.strip():
            st.warning("Please enter both a topic and target audience.")

        else:
            rows = []

            for v in VARIANTS:
                p = build(v, topic, aud)

                try:
                    ans, _ = call_llm(p)
                except Exception as e:
                    st.error(f"Unable to generate response: {e}")
                    st.stop()

                sc = judge(ans, topic, aud)

                with st.expander(
                    f"✨ {v.split('_')[1].title()} Prompt",
                    expanded=True
                ):
                    st.markdown("**Generated Prompt**")
                    st.code(p.replace("[[LADDER]] ", ""))

                    st.markdown("**AI Response**")
                    st.write(ans)

                    st.metric("Rubric Score / 20", sum(sc.values()))

                rows.append(
                    dict(variant=v, **sc, total=sum(sc.values()))
                )

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.bar_chart(pd.DataFrame(rows).set_index("variant")["total"])

with tab2:
    st.header("⚖️ Technique Comparator")

    st.caption(
        "Compare different prompt engineering techniques "
        "on the same input and task."
    )

    st.markdown("### 1. Select your task")

    task = st.selectbox(
        "Task type",
        list(TASK_INSTRUCTIONS)
    )

    st.markdown("### 2. Enter your content")

    text = st.text_area(
        "Your input",
        height=160,
        placeholder="Enter your text or question here..."
    )

    techs = st.multiselect(
        "Techniques",
        list(TECHNIQUES),
        default=list(TECHNIQUES)
    )

    if st.button("Compare", use_container_width=True):

        if not text.strip():
            st.warning("Please enter some text or a question before comparing.")

        elif not techs:
            st.warning("Please select at least one technique.")

        else:
            rows = []

            for t in techs:
                prompt = TECHNIQUES[t](task, text)

                try:
                    out, lat = call_llm(prompt)
                except Exception as e:
                    st.error(f"Unable to generate response: {e}")
                    st.stop()

                with st.expander(f"✨ {t}", expanded=True):

                    st.markdown("**Generated Prompt**")
                    st.code(prompt)

                    st.markdown("**AI Response**")
                    st.write(out)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Prompt Tokens", approx_tokens(prompt))

                    with col2:
                        st.metric("Latency", f"{lat:.2f} sec")

                rows.append(dict(
                    technique=t,
                    answer=extract_answer(task, out)[:80],
                    prompt_tokens=approx_tokens(prompt),
                    latency_s=round(lat, 2)
                ))

            st.divider()
            st.subheader("📊 Comparison Summary")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True
            )
with tab3:
    st.header("📊 Benchmark Results")
    st.caption("Visual comparison of prompt engineering performance.")

    # Dashboard Overview Cards
    try:
        summary = pd.read_csv("results/summary.csv")

        st.subheader("📌 Performance Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_score = summary["score"].mean()
            st.metric("Average Quality Score", f"{avg_score:.2f}")

        with col2:
            avg_consistency = summary["consistency"].mean()
            st.metric("Average Consistency", f"{avg_consistency:.2f}")

        with col3:
            avg_tokens = summary["cost_tokens"].mean()
            st.metric("Average Token Usage", f"{avg_tokens:.1f}")

        st.divider()

    except (FileNotFoundError, KeyError):
        st.info("Run the benchmark to display performance metrics.")

    # Benchmark tables
    for f, label in [
        ("results/summary.csv", "Technique Benchmark"),
        ("results/ladder_summary.csv", "Prompt Ladder")
    ]:
        st.subheader(label)

        try:
            df = pd.read_csv(f)

            st.dataframe(
                df.style
                .format(precision=2)
                .set_properties(**{
                    "background-color": "#151F35",
                    "color": "#F1F5F9",
                    "border-color": "#34415F",
                    "font-size": "16px",
                    "font-family": "Times New Roman"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#312E81"),
                            ("color", "white"),
                            ("font-size", "17px"),
                            ("font-family", "Times New Roman")
                        ]
                    }
                ]),
                use_container_width=True
            )

        except FileNotFoundError:
            st.info(f"Run the matching script to create {f}.")

    st.divider()

    # Benchmark visualizations
    st.subheader("📈 Performance Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Technique Score Comparison**")
        try:
            st.image("results/score.png", use_container_width=True)
        except FileNotFoundError:
            st.info("score.png not found.")

    with col2:
        st.markdown("**Token Cost Comparison**")
        try:
            st.image("results/cost_tokens.png", use_container_width=True)
        except FileNotFoundError:
            st.info("cost_tokens.png not found.")

    st.markdown("**Consistency Comparison**")
    try:
        st.image("results/consistency.png", use_container_width=True)
    except FileNotFoundError:
        st.info("consistency.png not found.")
