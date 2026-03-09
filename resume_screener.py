"""
========================================
  AI RESUME SCREENING SYSTEM
  Using NLP + Scikit-learn
========================================
"""

import re
import json
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ──────────────────────────────────────────────
# 1. SKILL TAXONOMY
# ──────────────────────────────────────────────

SKILL_TAXONOMY = {
    "programming": [
        "python","java","javascript","typescript","c++","c#","golang","rust",
        "kotlin","swift","php","ruby","scala","r","matlab","perl","bash","shell"
    ],
    "web": [
        "react","angular","vue","nextjs","nodejs","django","flask","fastapi",
        "html","css","sass","tailwind","bootstrap","jquery","graphql","rest api"
    ],
    "data": [
        "machine learning","deep learning","nlp","computer vision","tensorflow",
        "pytorch","keras","scikit-learn","pandas","numpy","matplotlib","seaborn",
        "tableau","power bi","data analysis","data science","statistics","sql"
    ],
    "cloud": [
        "aws","azure","gcp","docker","kubernetes","terraform","ci/cd","devops",
        "linux","git","jenkins","ansible","prometheus","grafana"
    ],
    "database": [
        "mysql","postgresql","mongodb","redis","elasticsearch","cassandra",
        "sqlite","oracle","dynamodb","firebase"
    ],
    "soft_skills": [
        "leadership","communication","teamwork","problem solving","agile",
        "scrum","project management","mentoring","collaboration","analytical"
    ],
}

ALL_SKILLS = {skill for skills in SKILL_TAXONOMY.values() for skill in skills}


# ──────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s\+\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> dict:
    """Extract skills from text and categorize them."""
    text_lower = text.lower()
    found = {cat: [] for cat in SKILL_TAXONOMY}

    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            # match whole word / phrase
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found[category].append(skill)

    return found


def extract_experience_years(text: str) -> int:
    """Try to extract total years of experience from resume text."""
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'experience\s+of\s+(\d+)\+?\s*years?',
    ]
    for pat in patterns:
        match = re.search(pat, text.lower())
        if match:
            return int(match.group(1))
    return 0


def extract_education(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["phd","ph.d","doctorate"]):
        return "PhD"
    if any(w in text_lower for w in ["master","msc","m.s","m.tech","mba"]):
        return "Masters"
    if any(w in text_lower for w in ["bachelor","bsc","b.s","b.tech","b.e","undergraduate"]):
        return "Bachelors"
    return "Not specified"


# ──────────────────────────────────────────────
# 3. SCORING ENGINE
# ──────────────────────────────────────────────

def score_resume(resume_text: str, job_description: str) -> dict:
    """
    Score a single resume against a job description.
    Returns a dict with score breakdown.
    """
    resume_clean = clean_text(resume_text)
    jd_clean     = clean_text(job_description)

    # ── TF-IDF cosine similarity ──────────────
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_clean, resume_clean])
        similarity   = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception:
        similarity = 0.0

    # ── Skill match ───────────────────────────
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(job_description)

    all_jd_skills     = [s for cat in jd_skills.values()  for s in cat]
    all_resume_skills = [s for cat in resume_skills.values() for s in cat]

    matched   = set(all_jd_skills) & set(all_resume_skills)
    skill_pct = (len(matched) / max(len(all_jd_skills), 1)) * 100

    # ── Experience ────────────────────────────
    exp_years = extract_experience_years(resume_text)
    exp_score = min(exp_years / 10 * 100, 100)   # cap at 10 yrs = 100

    # ── Education bonus ───────────────────────
    edu        = extract_education(resume_text)
    edu_bonus  = {"PhD": 15, "Masters": 10, "Bachelors": 5}.get(edu, 0)

    # ── Composite score (weighted) ────────────
    final_score = (
        similarity  * 40 +   # text relevance
        skill_pct   * 0.35 + # skill match
        exp_score   * 0.15 + # experience
        edu_bonus          # education
    )
    final_score = min(round(final_score, 1), 100)

    return {
        "similarity_score" : round(similarity * 100, 1),
        "skill_match_pct"  : round(skill_pct, 1),
        "matched_skills"   : sorted(matched),
        "all_resume_skills": all_resume_skills,
        "experience_years" : exp_years,
        "education"        : edu,
        "final_score"      : final_score,
        "skill_breakdown"  : resume_skills,
    }


# ──────────────────────────────────────────────
# 4. RANK MULTIPLE CANDIDATES
# ──────────────────────────────────────────────

def rank_candidates(resumes: list[dict], job_description: str) -> pd.DataFrame:
    """
    resumes: list of {"name": str, "text": str}
    Returns DataFrame sorted by final_score descending.
    """
    results = []
    for r in resumes:
        scored = score_resume(r["text"], job_description)
        results.append({
            "Rank"            : 0,
            "Candidate"       : r["name"],
            "Final Score"     : scored["final_score"],
            "Text Similarity" : scored["similarity_score"],
            "Skill Match %"   : scored["skill_match_pct"],
            "Matched Skills"  : ", ".join(scored["matched_skills"]) or "None",
            "Experience (yrs)": scored["experience_years"],
            "Education"       : scored["education"],
        })

    df = pd.DataFrame(results).sort_values("Final Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


# ──────────────────────────────────────────────
# 5. SAMPLE DATA
# ──────────────────────────────────────────────

SAMPLE_JD = """
We are looking for a Senior Data Scientist with strong Python skills.
Requirements:
- 4+ years of experience in machine learning and data science
- Proficiency in Python, TensorFlow or PyTorch, scikit-learn
- Experience with SQL, pandas, numpy, matplotlib
- Knowledge of NLP and deep learning techniques
- Familiarity with AWS or GCP cloud platforms
- Strong communication and teamwork skills
- Masters or PhD preferred
"""

SAMPLE_RESUMES = [
    {
        "name": "Alice Johnson",
        "text": """
        Senior Data Scientist with 6 years of experience.
        Skills: Python, TensorFlow, PyTorch, scikit-learn, NLP, deep learning,
        pandas, numpy, SQL, AWS, communication, teamwork, leadership.
        Masters in Computer Science from MIT.
        Built ML pipelines, deployed models on AWS, led team of 4 engineers.
        """
    },
    {
        "name": "Bob Martinez",
        "text": """
        Data Analyst with 3 years of experience in data analysis and SQL.
        Proficient in Python, pandas, matplotlib, Power BI, Excel.
        Bachelor in Statistics. Good at communication and problem solving.
        Limited experience in machine learning. Some exposure to scikit-learn.
        """
    },
    {
        "name": "Carol Chen",
        "text": """
        Machine Learning Engineer, 5 years experience.
        Expert in Python, scikit-learn, TensorFlow, Keras, NLP, computer vision.
        Used AWS, Docker, Kubernetes for deployment. PhD in AI.
        Strong analytical and collaboration skills. SQL, MongoDB experience.
        """
    },
    {
        "name": "David Kumar",
        "text": """
        Full Stack Developer, 4 years experience.
        JavaScript, React, Node.js, Python (basic), HTML, CSS, MySQL.
        Bachelor in Computer Science. Good teamwork. Some REST API experience.
        No machine learning background. Learning Python data science recently.
        """
    },
    {
        "name": "Eva Rossi",
        "text": """
        Data Scientist, 4 years experience. Python, pandas, numpy, scikit-learn,
        matplotlib, seaborn, SQL, Tableau. Masters in Data Science.
        NLP projects using spacy and NLTK. GCP certified. Agile environment.
        Strong communication. Presented findings to C-suite stakeholders.
        """
    },
]


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖  AI RESUME SCREENING SYSTEM")
    print("=" * 60)

    print("\n📋  JOB DESCRIPTION:")
    print(SAMPLE_JD)

    print("\n⚙️   Analyzing candidates...\n")
    df = rank_candidates(SAMPLE_RESUMES, SAMPLE_JD)

    print("🏆  CANDIDATE RANKINGS")
    print("=" * 60)
    print(df[["Rank","Candidate","Final Score","Skill Match %","Experience (yrs)","Education"]].to_string(index=False))

    print("\n\n📊  DETAILED BREAKDOWN")
    print("=" * 60)
    for _, row in df.iterrows():
        medal = ["🥇","🥈","🥉","4️⃣ ","5️⃣ "][int(row["Rank"])-1]
        print(f"\n{medal}  {row['Candidate']}  —  Score: {row['Final Score']}/100")
        print(f"   Text Match   : {row['Text Similarity']}%")
        print(f"   Skill Match  : {row['Skill Match %']}%")
        print(f"   Experience   : {row['Experience (yrs)']} years")
        print(f"   Education    : {row['Education']}")
        print(f"   Key Skills   : {row['Matched Skills']}")

    print("\n\n💬  INTERACTIVE MODE — paste a resume and job description")
    print("=" * 60)
    print("Commands:  'jd'=set job description  'resume'=add resume  'rank'=show rankings  'quit'=exit\n")

    custom_jd       = SAMPLE_JD
    custom_resumes  = []

    while True:
        cmd = input("Command > ").strip().lower()
        if cmd in ("quit","exit","q"):
            print("Bye! 👋")
            break
        elif cmd == "jd":
            print("Paste job description (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            custom_jd = "\n".join(lines)
            print("✅ Job description set.")
        elif cmd == "resume":
            name = input("Candidate name: ").strip()
            print("Paste resume text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            custom_resumes.append({"name": name, "text": "\n".join(lines)})
            print(f"✅ Resume added for {name}.")
        elif cmd == "rank":
            if not custom_resumes:
                print("No resumes added yet. Use 'resume' command first.")
                continue
            df2 = rank_candidates(custom_resumes, custom_jd)
            print("\n🏆  RANKINGS")
            print(df2[["Rank","Candidate","Final Score","Skill Match %","Education"]].to_string(index=False))
        else:
            print("Unknown command. Use: jd / resume / rank / quit")
