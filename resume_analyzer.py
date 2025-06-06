import streamlit as st
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer
import numpy as np

# Set page config FIRST before any Streamlit commands
st.set_page_config(page_title="Resume Score", layout="centered")

# Load model using sentence-transformers
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    if uploaded_file is None:
        return ""
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Streamlit UI
st.title("📊 AI-Powered Resume Analyzer")

# Job Description Input
jd_input_type = st.radio("Input JD as:", ["Text Input", "Upload Word (.docx)"])
if jd_input_type == "Text Input":
    job_description = st.text_area("Enter Job Description", height=200)
else:
    jd_file = st.file_uploader("Upload JD (.docx)", type=["docx"])
    job_description = extract_text_from_docx(jd_file) if jd_file else ""

# Resume Upload
resume_file = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf", "docx"])
resume_text = ""
if resume_file:
    if resume_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_file)
    elif resume_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(resume_file)

# Run match
if st.button("🔍 Compute Match Score"):
    if not job_description.strip() or not resume_text.strip():
        st.warning("Please upload both a resume and a job description.")
    else:
        with st.spinner("Computing..."):
            jd_embed = model.encode([job_description])[0]
            resume_embed = model.encode([resume_text])[0]
            # Compute cosine similarity
            similarity = np.dot(jd_embed, resume_embed) / (np.linalg.norm(jd_embed) * np.linalg.norm(resume_embed))
            score = similarity * 100
            st.metric(label="Match Score", value=f"{score:.2f}/100")


