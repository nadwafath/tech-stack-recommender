Tech Stack Recommender (Content-Based Filtering) 🎯

A recommendation engine built as Project 3 of the DecodeLabs Artificial Intelligence Industrial Training track. It takes a user's skills as input and recommends the best-matching tech job roles using content-based filtering — no user history required.

Overview

Given at least 3 skills (e.g. Python, Cloud, Automation), this engine compares them against a dataset of job roles and their required skills, then returns the Top 3 best-matching roles with a match percentage for each — turning a simple list of skills into ranked, data-driven career suggestions.

How It Works

This uses content-based filtering: matching a user's profile directly to item (job role) attributes, rather than relying on other users' behavior (collaborative filtering). This means it works instantly for any new user, with no "cold start" problem.

Vectorization (TF-IDF) — Every job role's skill list, plus the user's own input, gets converted into numeric vectors. Skills that appear in few job roles are weighted more heavily (they're distinctive), while skills common across almost every role are weighted down (they're less useful for telling roles apart).
Similarity Scoring (Cosine Similarity) — Measures the angle between the user's skill vector and each job's skill vector, producing a 0–100% match score that isn't distorted by differing list lengths.
Ranking & Filtering — Job roles are sorted by match score, and only the top 3 are returned, avoiding "choice overload."

Features
🧑‍💻 Interactive terminal input — type in your own skills, comma-separated
📊 15 pre-loaded job roles spanning data, cloud, security, dev, and more
🎯 TF-IDF + Cosine Similarity for genuinely meaningful matching (not just keyword counting)
🏆 Ranked Top-3 output with match percentages

Getting Started
Prerequisites
Python 3.9+
pandas, scikit-learn
Install dependencies
bash
pip install pandas scikit-learn
Run it
bash
python Project3.py

You'll be prompted to enter your skills — try something like:

Python, Cloud, Automation
Example Output
#1: Network Engineer  (match: 45.02%)
    Required skills: Networking, Security, Cloud, Linux, Firewalls, Automation

#2: Systems Administrator  (match: 43.29%)
    Required skills: Linux, Networking, Automation, Cloud, Security, Scripting

#3: Cloud Architect  (match: 36.38%)
    Required skills: AWS, Azure, Cloud, Networking, Security, Automation, Infrastructure
Files
tech-stack-recommender/
├── Project3.py         # Main recommendation engine
├── raw_skills.csv       # Job roles dataset (15 roles + required skills)
└── README.md
Built With
Python
pandas — dataset handling
scikit-learn — TF-IDF vectorization and cosine similarity
Limitations
Cold Start Problem: skills not present anywhere in the dataset won't contribute to any match
Content-based only — doesn't learn from other users' choices over time (that would require collaborative filtering, a further extension)
Roadmap / Possible Extensions
Expand the job roles dataset with more roles and skills
Add a simple GUI or web form for input instead of terminal typing
Blend in collaborative filtering once real user interaction data exists
Let users rate recommendations to improve future matches
Author

Built as part of the DecodeLabs Artificial Intelligence Industrial Training Kit (Batch 2026).

License

This project is open source and available under the MIT License.
