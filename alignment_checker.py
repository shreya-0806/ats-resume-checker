# alignment_checker.py
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from cohere import Client

lemmatizer = WordNetLemmatizer()
co = Client(os.getenv("COHERE_API_KEY"))

def get_alignment_score(resume_text, job_desc_text):
    # Tokenize and lemmatize
    resume_tokens = [lemmatizer.lemmatize(w.lower()) for w in re.findall(r'\b\w+\b', resume_text)]
    jd_tokens = [lemmatizer.lemmatize(w.lower()) for w in re.findall(r'\b\w+\b', job_desc_text)]

    # Identify relevant job role keywords (length > 3 avoids noise)
    important_jd_keywords = [word for word in jd_tokens if len(word) > 3 and word in resume_tokens]
    matched_keywords = set(important_jd_keywords) & set(resume_tokens)

    if important_jd_keywords:
        keyword_score = len(matched_keywords) / len(set(important_jd_keywords)) * 100
    else:
        keyword_score = 0

    # Semantic similarity
    resp = co.embed(texts=[resume_text, job_desc_text], input_type="search_query")
    sem_score = 0
    if resp.embeddings and len(resp.embeddings) == 2:
        r, j = resp.embeddings
        dot = sum(a * b for a, b in zip(r, j))
        norm_r = sum(a * a for a in r) ** 0.5
        norm_j = sum(b * b for b in j) ** 0.5
        sem_score = (dot / (norm_r * norm_j)) * 100

    # Weighted combination
    final_score = 0.7 * sem_score + 0.3 * keyword_score
    return round(final_score, 2)
