import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import cohere
import nltk
nltk.data.path.append("nltk_data")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
import re
import docx
import pytesseract
from PIL import Image
from pdfplumber import open as pdfplumber_open

from section_detection import detect_sections
from alignment_checker import get_alignment_score
from keywords_by_course import TECH_KEYWORDS_BY_COURSE, SOFT_SKILLS_BY_COURSE, MULTIWORD_KEYWORDS_BY_COURSE, GENERIC_KEYWORDS

# Initial setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
co = cohere.Client(os.getenv("COHERE_API_KEY"))
st.set_page_config(page_title="ATS Resume Checker", page_icon="📄")
st.title("📄 ATS Resume Checker (Enhanced)")

# --- Course Selection ---
st.markdown("👋 Start by selecting your course or domain to get tailored keyword matching and suggestions.")
course_options = list(TECH_KEYWORDS_BY_COURSE.keys())
selected_course = st.selectbox(
    "🎓 Select Your Course / Domain",
    ["-- Select Course --"] + course_options,
    format_func=lambda x: x.replace("_", " ").title() if x != "-- Select Course --" else x
)

if selected_course == "-- Select Course --":
    st.warning("⚠️ Please select your course/domain before proceeding.")
    st.stop()

tech_keywords = TECH_KEYWORDS_BY_COURSE.get(selected_course, [])
soft_skills = SOFT_SKILLS_BY_COURSE.get(selected_course, [])
multiword_phrases = MULTIWORD_KEYWORDS_BY_COURSE.get(selected_course, [])
GENERIC_WORDS = GENERIC_KEYWORDS.get(selected_course, set())

# --- Helpers ---
def extract_text_from_pdf(file):
    try:
        with pdfplumber_open(file) as pdf:
            text = " ".join(page.extract_text() or "" for page in pdf.pages)
        if not text.strip():
            raise ValueError("No text extracted, try OCR fallback")
        return text
    except:
        try:
            image = Image.open(file)
            return pytesseract.image_to_string(image)
        except:
            return ""

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join(para.text for para in doc.paragraphs)

def clean_tokens(text):
    stop = set(stopwords.words("english"))
    tokens = re.findall(r'\b\w+\b', text.lower())
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop]
    return filtered, " ".join(filtered)

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

def match_skills(skills_list, text_tokens, text_full, threshold=85):
    matched = []
    for skill in skills_list:
        lemmatized = " ".join([lemmatizer.lemmatize(w) for w in skill.lower().split()])
        if lemmatized in text_full:
            matched.append(skill)
            continue
        for word in text_tokens:
            if fuzz.partial_ratio(skill.lower(), word.lower()) >= threshold:
                matched.append(skill)
                break
    return sorted(set(matched))

def get_resume_suggestions_cohere(text):
    prompt = (
        "You are an ATS optimization expert. Below is a resume. Give only 3 concise bullet points to improve it. "
        "Focus on adding missing keywords, section improvements, or formatting. Do NOT rewrite the resume.\n\n"
        f"{text}\n\n"
        "Your suggestions:"
    )
    resp = co.chat(model="command-r-plus", message=prompt, max_tokens=300, temperature=0.7)
    return resp.text.strip()

def get_semantic_similarity(resume, jobdesc):
    resp = co.embed(texts=[resume, jobdesc], input_type="search_query")
    if resp.embeddings and len(resp.embeddings) == 2:
        r, j = resp.embeddings
        dot = sum(a * b for a, b in zip(r, j))
        norm_r = sum(a * a for a in r) ** 0.5
        norm_j = sum(b * b for b in j) ** 0.5
        return round((dot / (norm_r * norm_j)) * 100, 2)
    return 0.0

# --- Main App ---
resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_desc_text = st.text_area("Paste Job Description Here", height=200, placeholder="Copy and paste the job description...")

if resume_file and job_desc_text and st.button("🔍 Check ATS Score"):
    resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
    resume_tokens, resume_cleaned = clean_tokens(resume_text)
    resume_joined = " ".join(resume_tokens)
    jd_tokens, jd_cleaned = clean_tokens(job_desc_text)

    important_keywords = set(kw.lower() for kw in tech_keywords + soft_skills + multiword_phrases)
    all_jd_skills = list(set(jd_tokens + multiword_phrases))
    scorable_keywords = [
        kw.lower() for kw in all_jd_skills
        if kw.lower() in important_keywords and kw.lower() not in GENERIC_WORDS and len(kw) > 2
    ]
    resume_token_set = set([lemmatizer.lemmatize(w.lower()) for w in resume_tokens])

    matched_keywords = []
    for skill in scorable_keywords:
        skill_clean = lemmatizer.lemmatize(skill.lower())
        if " " in skill_clean:
            if skill_clean in resume_text.lower():
                matched_keywords.append(skill_clean)
        else:
            for word in resume_token_set:
                if skill_clean == word or fuzz.ratio(skill_clean, word) >= 90:
                    matched_keywords.append(skill_clean)
                    break

    match_percent = round((len(set(matched_keywords)) / len(scorable_keywords)) * 100, 2) if scorable_keywords else 0

    tech_found = match_skills(tech_keywords, resume_tokens, resume_joined)
    soft_found = match_skills(soft_skills, resume_tokens, resume_joined)

    st.info("💬 Generating suggestions using Cohere…")
    suggestions = get_resume_suggestions_cohere(resume_text)
    st.info("🤔 Calculating semantic similarity…")
    semantic_score = get_semantic_similarity(resume_text, job_desc_text)
    st.info("🧹 Detecting sections…")
    found_sections, missing_sections = detect_sections(resume_text)
    st.info("🌟 Checking alignment with job role…")
    alignment_score = get_alignment_score(resume_text, job_desc_text)

    st.markdown("### 📊 Matching Summary")
    st.success(f"✅ ATS Match Score\n\n{match_percent:.2f}%")
    st.success(f"🤔 Semantic Similarity\n\n{semantic_score:.2f}%")
    st.success(f"🌟 Job Role Alignment Score\n\n{alignment_score:.2f}%")

    st.markdown("### ✅ Matched Keywords")
    st.write(", ".join(sorted(set(matched_keywords))) or "None")

    st.markdown("### ❌ Missing Keywords")
    missing_keywords = sorted(set(scorable_keywords) - set(matched_keywords))
    st.write(", ".join(missing_keywords) or "None")

    st.markdown("### 💻 Technical Skills")
    st.write(", ".join(tech_found) or "None")

    st.markdown("### 🤔 Soft Skills")
    st.write(", ".join(soft_found) or "None")

    st.markdown("### 🧹 Resume Section Detection")
    st.success(f"✅ Found Sections: {', '.join(found_sections).title() or 'None'}")
    st.error(f"❌ Missing Sections: {', '.join(missing_sections).title() or 'None'}")

    st.markdown("### ✨ Suggestions")
    st.write(suggestions)

else:
    st.warning("Please upload a resume and paste a job description to check the ATS score.")

# Footer
st.markdown("---")
st.markdown("Contact: smartfresherhubsa@gmail.com | Phone: +91 8299142475 © 2025 Smart Fresher Hub")
st.markdown("This app is powered by [Cohere](https://cohere.com) for AI suggestions and analysis.")
st.markdown("Made by Smart Fresher Hub")
st.markdown("**Disclaimer:** This tool is for educational purposes only and does not guarantee job placement or success. Always tailor your resume to the specific job you're applying for.")
