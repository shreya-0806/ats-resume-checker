# Updated keywords_by_course.py with more QA and Data Science keywords

TECH_KEYWORDS_BY_COURSE = {
    "programming": [
        "python", "java", "c++", "html", "css", "javascript", "react", "django", "flask",
        "api development", "git", "github", "oop", "web development", "software engineering",
              "selenium", "manual testing", "automation testing", "regression testing", "test cases",
           "test scenario", "xpath", "css selectors", "bug tracking", "jira", "webdriver",
        "defect tracking", "functional testing", "test script", "github", "version control",
        "test plan", "integration testing", "qa reporting"
    ],
    "data_analytics": [
        "data analysis", "data visualization", "machine learning", "python", "excel",
        "statistical analysis", "power bi", "tableau", "sql",
        "pandas", "numpy", "matplotlib", "jupyter notebook", "regression"
    ],
    "cybersecurity": [
        "network security", "ethical hacking", "firewall", "penetration testing", "malware analysis",
        "vulnerability assessment", "linux", "risk management"
    ],
    "civil_engineering": [
        "autocad", "revit", "sketchup", "construction planning", "building design", "project management"
    ],
    "mechanical_engineering": [
        "solidworks", "ansys", "autocad", "thermal analysis", "design thinking", "cad"
    ],
    "pharma": [
        "clinical research", "pharmacovigilance", "biostatistics", "regulatory compliance",
        "gmp", "sops", "clinical trials"
    ],
    "design": [
        "graphic design", "ux research", "ui design", "adobe photoshop", "illustrator", "branding",
        "figma", "canva", "motion graphics", "color theory", "visual hierarchy"
    ],
    "hr": [
        "recruitment", "performance appraisal", "employee engagement", "onboarding", "talent acquisition",
        "hr analytics", "conflict resolution", "training and development"
    ],
    "finance": [
        "financial analysis", "budgeting", "investment management", "financial reporting",
        "accounting principles", "taxation", "auditing", "excel", "cost accounting"
    ],
    "default": []
}

SOFT_SKILLS_BY_COURSE = {
    "programming": ["problem-solving", "communication", "teamwork", "adaptability"],
    "data_analytics": ["critical thinking", "attention to detail", "presentation skills"],
    "qa_testing": ["attention to detail", "logical thinking", "collaboration", "documentation"],
    "cybersecurity": ["integrity", "vigilance", "analytical thinking"],
    "civil_engineering": ["project coordination", "teamwork", "time management"],
    "mechanical_engineering": ["troubleshooting", "collaboration", "attention to detail"],
    "pharma": ["research skills", "documentation", "precision", "ethics"],
    "design": ["creativity", "aesthetic sense", "team collaboration", "user empathy"],
    "hr": ["emotional intelligence", "communication", "decision making", "teamwork"],
    "finance": ["analytical thinking", "problem solving", "attention to detail", "strategic thinking"],
    "default": ["communication", "teamwork", "leadership", "time management"]
}

MULTIWORD_KEYWORDS_BY_COURSE = {
    "programming": ["object oriented programming", "version control", "responsive design"],
    "data_analytics": ["data mining", "predictive modeling", "statistical analysis"],
    "qa_testing": ["test automation", "bug tracking", "quality assurance", "manual testing", "automation framework"],
    "cybersecurity": ["penetration testing", "incident response", "security audit"],
    "civil_engineering": ["construction project", "structural analysis", "building planning"],
    "mechanical_engineering": ["thermal systems", "finite element analysis", "machine design"],
    "pharma": ["clinical data management", "regulatory documentation", "pharmacovigilance reporting"],
    "design": ["user centered design", "brand identity", "design thinking"],
    "hr": ["talent acquisition", "employee relations", "organizational behavior"],
    "finance": ["investment portfolio", "financial modeling", "risk management"],
    "default": ["project management", "problem solving", "critical thinking"]
}

GENERIC_KEYWORDS = {
    "default": {"includes", "meet", "fully", "friendly", "standard", "creating", "additionally", "ensure", "using", "perform"},
    "programming": {"code", "coding", "debug"},
    "data_analytics": {"report", "reporting", "insights"},
    "qa_testing": {"tool", "testing", "test", "software"},
    "pharma": {"health", "patient", "medicine"},
    "civil_engineering": {"plan", "structure"},
    "mechanical_engineering": {"system", "design"},
    "cybersecurity": {"monitor", "detect"},
    "design": {"visual", "media", "creative"},
    "hr": {"process", "people", "employee"},
    "finance": {"money", "finance", "record"}
}
