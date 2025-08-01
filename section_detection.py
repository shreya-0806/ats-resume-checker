def detect_sections(text):
    section_map = {
        "education": ["education", "academic background"],
        "experience": ["experience", "work history", "internship"],
        "projects": ["projects", "technical projects"],
        "certifications": ["certifications", "certified", "certification", "course"],
        "skills": ["skills", "technical skills"],
        "objective": ["objective", "career objective", "summary"],
        "awards": ["awards", "achievements"],
        "publications": ["publications", "research"],
        "volunteer": ["volunteer", "community service"],
        "languages": ["languages", "language skills"]
    }

    text = text.lower()
    found = []
    for section, variants in section_map.items():
        if any(variant in text for variant in variants):
            found.append(section)

    missing = list(set(section_map.keys()) - set(found))
    return found, missing
