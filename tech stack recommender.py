"""DecodeLabs Project 3: AI Recommendation Logic — Tech Stack Recommender

Spec checklist covered:
- INPUT      -> Accepts a minimum of 3 user-provided skills
- PROCESS    -> Converts skills (user + job roles) into numeric vectors using TF-IDF,
                then measures similarity using Cosine Similarity (the industry-standard
                approach, since it isn't distorted by different text lengths the way
                raw Euclidean distance would be)
- OUTPUT     -> Returns a ranked Top-3 list of best-matching job roles ("Filtering" step,
                to avoid overwhelming the user with the full list — the "Choice Overload"
                problem the deck describes)

This is CONTENT-BASED filtering (matching user skills directly to item/job attributes),
not collaborative filtering (which needs other users' behavior data) — the deck
specifically calls out content-based as the right approach here, since it works
immediately without needing a big history of past users (avoids the "Cold Start" problem).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# STEP 1: INGESTION — load the job roles dataset (the "items" in this engine)
# ---------------------------------------------------------------------------
jobs = pd.read_csv("raw_skills.csv")

print("=" * 60)
print("STEP 1: JOB ROLE DATASET LOADED")
print("=" * 60)
print(f"Total job roles available: {len(jobs)}")
print(jobs[['job_role']].to_string(index=False))
print()


def recommend_jobs(user_skills, top_n=3):
    """
    Given a list of user skills, return the top_n best-matching job roles.

    user_skills: list of strings, e.g. ["Python", "Cloud", "Automation"]
    top_n: how many recommendations to return
    """
    # -----------------------------------------------------------------------
    # STEP 2: BRIDGING THE LANGUAGE BARRIER — turn text into numbers
    #
    # TF-IDF (Term Frequency - Inverse Document Frequency) converts each job's
    # skill list into a row of numbers, where:
    #   - a skill that appears in FEW job roles gets a HIGH weight (it's specific
    #     and meaningful, e.g. "Kubernetes")
    #   - a skill that appears in almost EVERY job role gets a LOW weight
    #     (it's too generic to be useful for telling roles apart, e.g. common words)
    #
    # We combine the user's own skills as one extra "document" so it gets
    # converted into the exact same numeric vocabulary as the job roles.
    # -----------------------------------------------------------------------
    user_profile_text = ", ".join(user_skills)
    all_documents = list(jobs['required_skills']) + [user_profile_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    # The last row is the user's profile vector; everything before it is job roles
    job_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]

    # -----------------------------------------------------------------------
    # STEP 3: SCORING — Cosine Similarity
    #
    # This measures the ANGLE between the user's vector and each job's vector,
    # not the raw distance. That matters because job descriptions vary in
    # length (some have 5 skills, some have 7) — cosine similarity ignores
    # that and focuses purely on how well the *pattern* of skills lines up.
    # Score ranges from 0 (no overlap at all) to 1 (perfect match).
    # -----------------------------------------------------------------------
    similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

    # -----------------------------------------------------------------------
    # STEP 4: SORTING & FILTERING — rank jobs best-to-worst, keep only top_n
    # -----------------------------------------------------------------------
    results = jobs.copy()
    results['match_score'] = similarity_scores
    results = results.sort_values(by='match_score', ascending=False)

    return results.head(top_n)[['job_role', 'required_skills', 'match_score']]


# ---------------------------------------------------------------------------
# DEMO — run the recommender with either your own input or an example profile
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: USER INPUT")
    print("=" * 60)
    print("Enter at least 3 skills, separated by commas.")
    print("Example: Python, Cloud, Automation")
    print()

    raw_input_text = input("Your skills: ").strip()

    if raw_input_text:
        my_skills = [skill.strip() for skill in raw_input_text.split(",") if skill.strip()]
    else:
        # fallback example if the user just presses Enter
        my_skills = ["Python", "Cloud", "Automation"]
        print(f"(No input given — using example skills: {my_skills})")

    if len(my_skills) < 3:
        print(f"\nNote: only {len(my_skills)} skill(s) entered. "
              f"The spec recommends at least 3 for a reliable match.")

    print(f"\nUser entered skills: {my_skills}")
    print()

    print("=" * 60)
    print("STEP 3 & 4: TOP 3 RECOMMENDED JOB ROLES")
    print("=" * 60)

    top_matches = recommend_jobs(my_skills, top_n=3)

    for rank, (_, row) in enumerate(top_matches.iterrows(), start=1):
        print(f"\n#{rank}: {row['job_role']}  (match: {row['match_score']:.2%})")
        print(f"    Required skills: {row['required_skills']}")