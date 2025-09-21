## AI recruitment analysis

This Streamlit app uses **OpenAI GPT-4o** and **LangChain** to analyze candidate resumes against job descriptions.  
It extracts structured information, generates summaries, and provides a **match score (0–100)** for recruitment suitability.

### Features
- Upload or auto-load resumes & job descriptions (`.txt` or `.pdf`).
- Extract key candidate information (education, skills, work history, etc.).
- Generate AI-powered summary of candidate suitability.
- Compute a **matching score** with scoring criteria.
- Download results as `.txt`.

### Installation

Clone the repository and requirements:
```bash
git clone https://github.com/werefin/Resume-Analyzer-Bot.git
cd Resume-Analyzer-Bot
pip install -r requirements.txt
