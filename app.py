import os
import streamlit as st
from dotenv import load_dotenv
import cohere
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import docx
from fpdf import FPDF
from io import BytesIO

load_dotenv()
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Skill keywords
tech_keywords = ['python', 'java', 'sql', 'c++', 'html', 'css', 'javascript', 'react', 'django', 'flask', 'aws']
soft_skills = ['teamwork', 'communication', 'leadership', 'problem-solving', 'adaptability', 'creativity']

# Helper Functions
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join(para.text for para in doc.paragraphs)

def clean_tokens(text):
    stop = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [w for w in tokens if w.isalnum() and w not in stop]

def get_resume_suggestions_cohere(text):
    prompt = (
        "You are a professional resume coach. "
        "Here is my resume text:\n\n"
        f"{text}\n\n"
        "Suggest improvements to make it more ATS-friendly. "
        "Include missing keywords, stronger wording, formatting tips, and skills suggestions."
    )
    resp = co.chat(
        model="command-r-plus",
        message=prompt,  # changed from 'messages' to 'message'
        max_tokens=400,
        temperature=0.7
    )
    return resp.text  # changed from resp.message.content[0].text

def generate_pdf_report(match_percent, matched, missing, tech_found, soft_found, suggestions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200,10,txt="ATS Resume Checker Report",ln=True,align='C')
    pdf.ln(10)
    pdf.cell(200,10,txt=f"Keyword Match: {match_percent:.2f}%",ln=True)
    pdf.ln(5)
    pdf.multi_cell(0,10, txt="Matched Keywords: " + (", ".join(sorted(matched)) or "None"))
    pdf.ln(5)
    pdf.multi_cell(0,10, txt="Missing Keywords: " + (", ".join(sorted(missing)) or "None"))
    pdf.ln(5)
    pdf.multi_cell(0,10, txt="Technical Skills Detected: " + (", ".join(sorted(tech_found)) or "None"))
    pdf.ln(5)
    pdf.multi_cell(0,10, txt="Soft Skills Detected: " + (", ".join(sorted(soft_found)) or "None"))
    pdf.ln(10)
    pdf.multi_cell(0,10, txt="Resume Suggestions:\n" + (suggestions or "None"))
    return BytesIO(pdf.output(dest='S').encode('latin1'))

# Streamlit UI
st.set_page_config(page_title="ATS Resume Checker", page_icon="📄")
st.title("📄 ATS Resume Checker (Paste Job Description)")

# Upload Resume
resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

# Paste Job Description
job_desc_text = st.text_area("Paste Job Description Here", height=200, placeholder="Copy and paste the job description...")

# Check ATS Match
if resume_file and job_desc_text and st.button("🔍 Check ATS Score"):
    # Extract text
    resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)

    # Token cleaning
    resume_tokens = clean_tokens(resume_text)
    jd_tokens = clean_tokens(job_desc_text)

    # Keyword match
    matched = set(resume_tokens) & set(jd_tokens)
    missing = set(jd_tokens) - set(resume_tokens)
    match_percent = (len(matched) / len(set(jd_tokens)) * 100) if jd_tokens else 0

    # Skills
    tech_found = [t for t in tech_keywords if t in resume_tokens]
    soft_found = [s for s in soft_skills if s in resume_tokens]

    # GPT Suggestions
    st.info("💬 Generating suggestions using Cohere…")
    suggestions = get_resume_suggestions_cohere(resume_text)

    # Results
    st.success(f"✅ ATS Match Score: {match_percent:.2f}%")
    st.markdown("### ✅ Matched Keywords")
    st.write(", ".join(sorted(matched)) or "None")
    st.markdown("### ❌ Missing Keywords")
    st.write(", ".join(sorted(missing)) or "None")
    st.markdown("### 💻 Technical Skills")
    st.write(", ".join(tech_found) or "None")
    st.markdown("### 🧠 Soft Skills")
    st.write(", ".join(soft_found) or "None")
    st.markdown("### ✨ Suggestions")
    st.write(suggestions)

    # Download Report
    pdf_bytes = generate_pdf_report(match_percent, matched, missing, tech_found, soft_found, suggestions)
    st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="ATS_Report.pdf", mime="application/pdf")
    st.success("Report generated successfully! You can download it using the button above.")    
else:
    st.warning("Please upload a resume and paste a job description to check the ATS score.")  
    # --footer--
    st.markdown("---")
    st.markdown("Contact: smartfresherhubsa@gmail.com | Phone: +91-8299142475 © 2025 Smart Fresher Hub | Build your career with confidence.")
# --- app.py ---
# This file contains the main application logic for the ATS Resume Checker.