import streamlit as st
import fitz
import docx
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="Resume Score", layout="centered")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc]).strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

st.title("📊 AI-Powered Resume Analyzer")

# Reset score when inputs change
def reset_score():
    st.session_state.pop("score", None)

# Job Description input
jd_input_type = st.radio("Input JD as:", ["Text Input", "Upload Word (.docx)"], on_change=reset_score)
if jd_input_type == "Text Input":
    job_description = st.text_area("Enter Job Description", height=200, key="jd_text", on_change=reset_score)
else:
    jd_file = st.file_uploader("Upload JD (.docx)", type=["docx"], key="jd_file", on_change=reset_score)
    job_description = extract_text_from_docx(jd_file) if jd_file else ""

# Resume upload
resume_file = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf", "docx"], key="resume_file", on_change=reset_score)
resume_text = ""
if resume_file:
    if resume_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_file)
    elif resume_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(resume_file)

# Warnings for empty content
if resume_file and not resume_text:
    st.warning("Resume appears to be empty or unreadable.")
if jd_input_type == "Upload Word (.docx)" and jd_file and not job_description:
    st.warning("Job Description appears to be empty or unreadable.")

# Compute score
if st.button("🔍 Compute Match Score"):
    if not job_description.strip() or not resume_text.strip():
        st.warning("Please upload both a resume and a job description.")
    else:
        jd_embed = model.encode([job_description])[0]
        resume_embed = model.encode([resume_text])[0]
        similarity = np.dot(jd_embed, resume_embed) / (np.linalg.norm(jd_embed) * np.linalg.norm(resume_embed))
        score = similarity * 100
        st.session_state["score"] = score  # Store the score

# Show score only if present
if "score" in st.session_state:
    score = st.session_state["score"]
    st.metric(label="Match Score", value=f"{score:.2f}/100")


# Clear button
if st.button("🔄 Clear"):
    st.session_state.clear()
    st.rerun()

