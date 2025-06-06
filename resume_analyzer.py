import streamlit as st
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer
import numpy as np

# Set page config FIRST
st.set_page_config(page_title="Resume Score", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    try:
        if uploaded_file is None:
            return ""
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

# Streamlit UI
st.title("📊 AI-Powered Resume Analyzer")

# Job Description Input
jd_input_type = st.radio("Input JD as:", ["Text Input", "Upload Word (.docx)"])
if jd_input_type == "Text Input":
    job_description = st.text_area("Enter Job Description", height=200)
else:
    jd_file = st.file_uploader("Upload JD (.docx)", type=["docx"], key="jd_file")
    job_description = extract_text_from_docx(jd_file) if jd_file else ""

# Resume Upload
resume_file = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf", "docx"], key="resume_file")
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

# Compute Score Button
if st.button("🔍 Compute Match Score"):
    if not job_description.strip() or not resume_text.strip():
        st.warning("Please upload both a resume and a job description.")
    else:
        # Save into session state
        st.session_state["job_description"] = job_description
        st.session_state["resume_text"] = resume_text

# Score Calculation
if "job_description" in st.session_state and "resume_text" in st.session_state:
    with st.spinner("Computing..."):
        jd_embed = model.encode([st.session_state["job_description"]])[0]
        resume_embed = model.encode([st.session_state["resume_text"]])[0]
        similarity = np.dot(jd_embed, resume_embed) / (
            np.linalg.norm(jd_embed) * np.linalg.norm(resume_embed)
        )
        score = similarity * 100
        st.metric(label="Match Score", value=f"{score:.2f}/100")

      
# Clear Button
if st.button("🔄 Clear"):
    st.session_state.clear()
    st.rerun()
