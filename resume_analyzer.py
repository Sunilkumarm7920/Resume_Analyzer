import streamlit as st
import pymupdf
import docx
import tensorflow_hub as hub
import tensorflow as tf

# Load model using TensorFlow Hub
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

model = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Streamlit UI
st.set_page_config(page_title="Resume Score", layout="centered")
st.title("📊 Resume Match Score")

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
            jd_embed = model([job_description])[0]
            resume_embed = model([resume_text])[0]
            score = tf.keras.losses.cosine_similarity(jd_embed, resume_embed).numpy()
            similarity = (1 + score) * 50  # convert from cosine distance to percentage similarity
            st.metric(label="Match Score", value=f"{similarity:.2f}/100")
