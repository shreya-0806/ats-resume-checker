import os
import streamlit as st
from dotenv import load_dotenv
import cohere
import nltk
nltk.data.path.append("nltk_data")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fpdf import FPDF
import docx
import re
from fpdf import FPDF
from io import BytesIO
from grammar_checker import check_grammar  # ⬅️ Grammar check
from section_detection import detect_sections  # ⬅️ Section detection
from alignment_checker import get_alignment_score  # ⬅️ Job role alignment
import docx
from pdfplumber import open as pdfplumber_open  # ⬅️ Better PDF parsing
import pytesseract
from PIL import Image
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz

lemmatizer = WordNetLemmatizer()

load_dotenv()
nltk.download('stopwords')
nltk.download('wordnet')  # <-- Add this line
nltk.download('omw-1.4')

# Initialize Cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Skill keywords
tech_keywords = ['python', 'java', 'sql', 'c++', 'html', 'css', 'javascript', 'react', 'django', 'flask', 'aws'
                 , 'azure', 'docker', 'kubernetes', 'git', 'linux', 'machine learning', 'data analysis'
                , 'big data', 'cloud computing', 'cybersecurity', 'networking', 'database management'
                , 'mobile development', 'web development', 'software engineering', 'agile methodology'
                , 'devops', 'api development', 'microservices', 'ui/ux design', 'responsive design'
                , 'test automation', 'quality assurance', 'performance optimization', 'scalability'
                , 'system architecture', 'data visualization', 'business intelligence', 'blockchain technology'
                , 'artificial intelligence', 'internet of things'
                , 'augmented reality', 'virtual reality', 'content management', 'digital marketing'
                , 'search engine optimization', 'social media marketing', 'email marketing', 'brand management'
                , 'public relations', 'event planning', 'sales strategy', 'customer relationship management'
                , 'lead generation', 'market research', 'business development', 'financial analysis'
                , 'budget management', 'supply chain management', 'logistics', 'operations management'
                , 'human resources', 'talent acquisition', 'employee engagement', 'performance appraisal'
                , 'training and development', 'organizational behavior', ]
soft_skills = ['teamwork', 'communication', 'leadership', 'problem-solving', 'adaptability', 'creativity'
               , 'time management', 'critical thinking', 'emotional intelligence', 'conflict resolution'
               , 'decision making', 'negotiation', 'collaboration', 'customer service', 'flexibility'
               , 'attention to detail', 'organizational skills', 'self-motivation', 'resilience'
               , 'interpersonal skills', 'active listening', 'positive attitude', 'work ethic'
               , 'stress management', 'cultural awareness', 'networking', 'public speaking'
               , 'mentoring', 'coaching', 'influence', 'persuasion', 'strategic thinking'
               , 'analytical thinking', 'innovation', 'visionary thinking', 'project management'
               , 'time management', 'goal setting', 'planning', 'execution', 'evaluation'
               , 'risk management', 'change management', 'resource management', 'stakeholder management'
               , 'process improvement', 'quality assurance', 'customer focus', 'business acumen'
               , 'financial literacy', 'data analysis', 'market research', 'sales skills'
               , 'negotiation skills', 'relationship building', 'empathy', 'trustworthiness'
               , 'confidentiality', 'professionalism', 'work-life balance', 'self-awareness'
               , 'self-regulation', 'motivation', 'goal orientation', 'results orientation'
               , 'initiative', 'proactivity', 'persistence', 'flexibility', 'adaptability'
               , 'creativity', 'innovation', 'vision', 'strategic planning', 'problem-solving'
               , 'decision-making', 'critical thinking', 'analytical skills', 'research skills'
               , 'data analysis', 'statistical analysis', 'quantitative skills', 'qualitative skills'
               , 'communication skills', 'verbal communication', 'written communication'
                , 'presentation skills', 'public speaking', 'listening skills', 'interpersonal skills'
                , 'negotiation skills', 'persuasion skills', 'influence skills', 'conflict resolution'
                , 'teamwork', 'collaboration', 'management skills', 'coaching skills'
                , 'mentoring skills', 'emotional intelligence', 'cultural awareness', 'diversity and inclusion'
                , 'customer service', 'sales skills', 'marketing skills', 'business development'
                , 'financial skills', 'budgeting skills', 'project management', 'time management'
                , 'planning skills', 'execution skills', 'evaluation skills'
                , 'risk management', 'change management', 'resource management', 'stakeholder management'
                , 'process improvement', 'quality assurance', 'customer focus', 'business acumen']
multiword_phrases = [
    "user interface", "user experience", "responsive design", 
    "frontend developer", "quality assurance", "test automation", 
    "cross-browser compatibility", "object oriented programming", 
    "web application", "api integration", "project management",
    "agile methodology", "version control", "continuous integration",
    "data analysis", "machine learning", "cloud computing",
    "database management", "network security", "mobile development",
    "software development", "technical writing", "business analysis",
    "customer service", "time management", "conflict resolution",
    "critical thinking", "decision making", "emotional intelligence",
    "strategic planning", "risk management", "change management",
    "performance optimization", "scalability", "system architecture",
    "data visualization", "big data", "internet of things",
    "artificial intelligence", "cybersecurity", "blockchain technology",
    "augmented reality", "virtual reality", "user research",
    "content management", "digital marketing", "search engine optimization",
    "social media marketing", "email marketing", "brand management",
    "public relations", "event planning", "sales strategy",
    "customer relationship management", "lead generation", "market research",
    "business development", "financial analysis", "budget management",
    "supply chain management", "logistics", "operations management",
    "human resources", "talent acquisition", "employee engagement",
    "performance appraisal", "training and development", "organizational behavior"
]

GENERIC_WORDS = {
    "includes", "meet", "fully", "friendly", "standard", "creating", "additionally", "ensure", "using", "perform", "hosted", "appealing", "visually"
}


# Helper Functions
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
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop]
    return filtered_tokens, " ".join(filtered_tokens)  # Return both tokens and cleaned full text

def fuzzy_skill_match(skill_list, resume_text, threshold=85):
    return sorted(set([
        skill for skill in skill_list 
        if any(fuzz.partial_ratio(skill.lower(), line.lower()) >= threshold 
               for line in resume_text.split('\n'))
    ]))


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
def get_semantic_similarity(resume, jobdesc):
    resp = co.embed(texts=[resume, jobdesc], input_type="search_query")
    if resp.embeddings and len(resp.embeddings) == 2:
        r, j = resp.embeddings
        dot = sum(a * b for a, b in zip(r, j))
        norm_r = sum(a * a for a in r) ** 0.5
        norm_j = sum(b * b for b in j) ** 0.5
        score = (dot / (norm_r * norm_j)) * 100
        return round(score, 2)
    return 0.0


def generate_pdf_report(match_percent, matched, missing, tech_found, soft_found, suggestions, found_sections, missing_sections, semantic_score, alignment_score, grammar_issues):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200,10,txt="ATS Resume Checker Report",ln=True,align='C')
    pdf.ln(10)

    pdf.cell(200,10,txt=f"Keyword Match: {match_percent:.2f}%",ln=True)
    pdf.cell(200,10,txt=f"Semantic Similarity: {semantic_score:.2f}%",ln=True)
    pdf.cell(200,10,txt=f"Job Role Alignment Score: {alignment_score:.2f}%",ln=True)

    pdf.ln(5)
    pdf.multi_cell(0,10, txt="Matched Keywords: " + (", ".join(sorted(matched)) or "None"))
    pdf.multi_cell(0,10, txt="Missing Keywords: " + (", ".join(sorted(missing)) or "None"))
    pdf.multi_cell(0,10, txt="Technical Skills Detected: " + (", ".join(sorted(tech_found)) or "None"))
    pdf.multi_cell(0,10, txt="Soft Skills Detected: " + (", ".join(sorted(soft_found)) or "None"))
    pdf.multi_cell(0,10, txt="Found Sections: " + (", ".join(found_sections).title() or "None"))
    pdf.multi_cell(0,10, txt="Missing Sections: " + (", ".join(missing_sections).title() or "None"))
    pdf.multi_cell(0,10, txt="Grammar Issues: \n" + (grammar_issues or "None"))

    pdf.ln(10)
    pdf.multi_cell(0,10, txt="Resume Suggestions:\n" + (suggestions or "None"))
    return BytesIO(pdf.output(dest='S').encode('latin1'))

# ---------------- Streamlit App ---------------- #

st.set_page_config(page_title="ATS Resume Checker", page_icon="📄")
st.title("📄 ATS Resume Checker (Enhanced)")

resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_desc_text = st.text_area("Paste Job Description Here", height=200, placeholder="Copy and paste the job description...")

if resume_file and job_desc_text and st.button("🔍 Check ATS Score"):
    resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
    resume_tokens, resume_cleaned = clean_tokens(resume_text)
    resume_joined = " ".join(resume_tokens)
    
    # Exact and fuzzy skill match (for keywords and phrases)
    def match_skills(skills_list, text_tokens, text_full, threshold=85):
     matched = []
     for skill in skills_list:
        lemmatized_skill = " ".join([lemmatizer.lemmatize(w) for w in skill.split()])
        if lemmatized_skill in text_full:
            matched.append(skill)
        else:
            # Try fuzzy match if exact match fails
            for word in text_tokens:
                if fuzz.ratio(skill.lower(), word.lower()) >= threshold:
                    matched.append(skill)
                    break
     return sorted(set(matched))

    # Lemmatize for better matching
    def lemmatize_tokens(tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]

    resume_lemmas = set(lemmatize_tokens(resume_tokens))
    # Clean and tokenize job description
    jd_tokens, jd_cleaned = clean_tokens(job_desc_text)
    jd_lemmas = set(lemmatize_tokens(jd_tokens))

    matched = resume_lemmas & jd_lemmas
    missing = jd_lemmas - resume_lemmas

    # Phrase match (multiword)
    resume_text_lower = resume_text.lower()
    resume_cleaned = clean_tokens(resume_text)[1]
    
    # Match against real skill sets
    tech_found = match_skills(tech_keywords, resume_tokens, resume_joined)
    soft_found = match_skills(soft_skills, resume_tokens, resume_joined)
    
    soft_found = sorted(set(soft_found))
    tech_found = sorted(set(tech_found))
    
    matched_phrases = match_skills(multiword_phrases, resume_tokens, resume_joined)

   # Calculate overall keyword match (JD → Resume)
    jd_tokens, jd_cleaned = clean_tokens(job_desc_text)

   # ✅ Filter only phrases that are present in JD
    relevant_phrases = [phrase for phrase in multiword_phrases if phrase in jd_cleaned]

# Combine JD tokens and relevant phrases
    all_jd_skills = list(set(jd_tokens + relevant_phrases))

    all_matched = match_skills(all_jd_skills, resume_tokens, resume_joined)
    all_missing = sorted(set(all_jd_skills) - set(all_matched))
    match_percent = (len(all_matched) / len(set(all_jd_skills)) * 100) if all_jd_skills else 0

    tech_found = [t for t in tech_keywords if t in resume_cleaned]
    soft_found = [s for s in soft_skills if s in resume_cleaned]


    st.info("💬 Generating suggestions using Cohere…")
    suggestions = get_resume_suggestions_cohere(resume_text)

    st.info("🧠 Calculating semantic similarity…")
    semantic_score = get_semantic_similarity(resume_text, job_desc_text)

    st.info("🧩 Detecting sections…")
    section_keywords = {
        "education": ["education", "academic background"],
        "experience": ["experience", "work history", "professional experience", "internship"],
        "projects": ["projects", "personal projects", "technical projects"],
        "certifications": ["certifications", "certifications & training", "courses"],
        "skills": ["skills", "technical skills"],
        "objective": ["objective", "career objective", "summary"],
        "awards": ["awards", "achievements", "honors"],
        "publications": ["publications", "research papers", "articles"],
        "volunteer": ["volunteer", "community service", "extracurricular"],
        "languages": ["languages", "language skills", "foreign languages"]
    }

    found_sections = []
    resume_lines = resume_text.lower().split('\n')
    for section, variants in section_keywords.items():
        if any(variant in line for variant in variants for line in resume_lines):
            found_sections.append(section)
    missing_sections = list(set(section_keywords.keys()) - set(found_sections))

    st.info("🎯 Checking alignment with job role…")
    alignment_score = get_alignment_score(resume_text, job_desc_text)

    st.info("✍️ Checking grammar…")
    grammar_issues = check_grammar(resume_text)

    st.success(f"✅ ATS Match Score: {match_percent:.2f}%")
    st.success(f"🧠 Semantic Similarity Score: {semantic_score:.2f}%")
    st.success(f"🎯 Job Role Alignment Score: {alignment_score:.2f}%")

    st.markdown("### ✅ Matched Keywords")
    st.write(", ".join(sorted(all_matched)) or "None")

    st.markdown("### ❌ Missing Keywords")
    # Filter out generic/common words from missing keywords
    important_keywords = set(tech_keywords + soft_skills + multiword_phrases)
    filtered_missing = [kw for kw in all_missing if kw in important_keywords and kw not in GENERIC_WORDS and len(kw) > 2]
    st.write(", ".join(sorted(filtered_missing)) or "None")

    st.markdown("### 💻 Technical Skills")
    st.write(", ".join(tech_found) or "None")

    st.markdown("### 🧠 Soft Skills")
    st.write(", ".join(soft_found) or "None")

    st.markdown("### 🧩 Resume Section Detection")
    st.success(f"✅ Found Sections: {', '.join(found_sections).title() or 'None'}")
    st.error(f"❌ Missing Sections: {', '.join(missing_sections).title() or 'None'}")

    st.markdown("### ✍️ Grammar Feedback")
    st.write(grammar_issues or "None")

    st.markdown("### ✨ Suggestions")
    st.write(suggestions)

    pdf_bytes = generate_pdf_report(match_percent, matched, missing, tech_found, soft_found, suggestions, found_sections, missing_sections, semantic_score, alignment_score, grammar_issues)
    st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="ATS_Report.pdf", mime="application/pdf")
    st.success("Report generated successfully! You can download it using the button above.")

else:
    st.warning("Please upload a resume and paste a job description to check the ATS score.")

st.markdown("---")
st.markdown("Contact: smartfresherhubsa@gmail.com | Phone: +91-8299142475 © 2025 Smart Fresher Hub | Build your career with confidence.")
st.markdown("Made by Smart Fresher Hub")
st.markdown("This app is powered by [Cohere](https://cohere.com) for AI suggestions and analysis.")
# Note: Ensure you have the required packages installed:
