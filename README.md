# HireIQ — AI Resume Screening System

> Automatically rank and screen resumes using NLP + Machine Learning + AI-powered web interface

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![HTML](https://img.shields.io/badge/UI-HTML%2FCSS%2FJS-red)
![AI](https://img.shields.io/badge/AI-Claude%20API-purple)

---

## 📌 Overview

HireIQ is a full-stack AI resume screening system that automatically ranks candidates against a job description. It extracts skills, calculates match scores, and provides AI-generated hiring recommendations — saving HR teams 10x screening time.

---

## 📁 Project Structure
```
AI-resume-screening-system/
├── resume_screener.py           # ML backend (Python + Scikit-learn)
├── resume_screener_website.html # Frontend website (single HTML file)
└── resume_requirements.txt      # Python dependencies
```

---

## ✨ Features

- 🎯 Automatically ranks candidates by match score (0–100)
- 🔍 Extracts 60+ skills across 6 categories (Programming, Web, Data, Cloud, Database, Soft Skills)
- 📊 TF-IDF cosine similarity for text relevance scoring
- 🎓 Detects education level (PhD / Masters / Bachelors)
- 📅 Extracts years of experience from resume text
- 🤖 AI-generated 2-sentence hiring recommendation per candidate
- 🌐 Single-file website — no server needed, opens in any browser
- 💻 Interactive CLI mode — paste resumes and rank live in terminal
- 4 sample job presets — Data Scientist, Frontend Dev, PM, ML Engineer

---

## 🚀 Quick Start

**1. Install dependencies**
```bash
pip install pandas numpy scikit-learn joblib
```

**2. Run the ML screener**
```bash
python resume_screener.py
```

**3. Open the website**

Double-click `resume_screener_website.html` in your browser — done!

---

## 🧠 How Scoring Works

| Component | Weight | Description |
|-----------|--------|-------------|
| TF-IDF Text Similarity | 40% | How closely resume matches JD language |
| Skill Match | 35% | % of required skills found in resume |
| Experience | 15% | Years of experience (capped at 10 yrs) |
| Education Bonus | +5–15 pts | Bachelors +5, Masters +10, PhD +15 |

**Final Score = Text(40%) + Skills(35%) + Experience(15%) + Education Bonus**

---

## 📊 Skill Categories Detected

| Category | Examples |
|----------|---------|
| Programming | Python, Java, JavaScript, TypeScript, C++, Go |
| Web | React, Angular, Vue, Node.js, Django, FastAPI |
| Data / ML | TensorFlow, PyTorch, scikit-learn, NLP, pandas |
| Cloud | AWS, Azure, GCP, Docker, Kubernetes |
| Database | MySQL, PostgreSQL, MongoDB, Redis |
| Soft Skills | Leadership, Agile, Communication, Scrum |

---

## 🌐 Website Features

- Paste any job description
- Add multiple candidates with their resume text
- One-click sample job presets (Data Scientist, Frontend, PM, ML Engineer)
- Animated score bars (Text Match, Skill Match, Overall Score)
- Green tags = skills matched to JD · Grey tags = extra resume skills
- 🤖 Claude AI writes a hiring note for each candidate

---

## ⚙️ Requirements
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

---

## 🛠️ Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'pandas'` | `pip install pandas numpy scikit-learn joblib` |
| `can't open file resume_screener.py` | Make sure you `cd` into the correct folder first |
| Yellow warnings in VS Code | Safe to ignore — just run `python resume_screener.py` in terminal |
| Website shows no AI notes | Check internet — Claude API used for summaries |

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ · HTML/CSS/JavaScript |
| ML | Scikit-learn — TF-IDF, Cosine Similarity |
| NLP | TF-IDF Vectorizer with unigram + bigram features |
| AI API | Anthropic Claude API (claude-sonnet-4-20250514) |
| Data | Pandas, NumPy |
| Fonts | Google Fonts: Clash Display, Cabinet Grotesk, DM Mono |

---

*Built by Kamali R · HireIQ AI Resume Screening System*
