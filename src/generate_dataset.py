"""Generate a synthetic student dataset (with realistic messiness) -> data/student_data.csv"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame(
    {
        "student_id": np.arange(1, n + 1),
        "gender": rng.choice(["Male", "Female"], n),
        "study_hours": np.clip(rng.normal(4, 2, n), 0, 12).round(1),
        "attendance": np.clip(rng.normal(80, 12, n), 35, 100).round(1),
        "previous_score": np.clip(rng.normal(65, 15, n), 20, 100).round(1),
        "sleep_hours": np.clip(rng.normal(7, 1.3, n), 3, 10).round(1),
        "assignments_completed": rng.integers(0, 11, n),
        "parental_education": rng.choice(
            ["High School", "Bachelor", "Master", "PhD"], n, p=[0.4, 0.35, 0.2, 0.05]
        ),
        "internet_access": rng.choice(["Yes", "No"], n, p=[0.85, 0.15]),
        "extracurricular": rng.choice(["Yes", "No"], n, p=[0.5, 0.5]),
        "tutoring": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    }
)

edu_bonus = df["parental_education"].map(
    {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
)
score = (
    8
    + 3.2 * df["study_hours"]
    + 0.25 * df["attendance"]
    + 0.35 * df["previous_score"]
    + 1.0 * (df["sleep_hours"] - 6).clip(-3, 2)
    + 1.2 * df["assignments_completed"]
    + 0.8 * edu_bonus
    + 2.5 * (df["tutoring"] == "Yes")
    + 1.5 * (df["internet_access"] == "Yes")
    + rng.normal(0, 4, n)
)
df["final_score"] = np.clip(score, 0, 100).round(1)

# --- add realistic mess: missing values + a few duplicates ---
for col in [
    "study_hours",
    "attendance",
    "sleep_hours",
    "parental_education",
    "previous_score",
]:
    df.loc[rng.choice(n, size=int(0.04 * n), replace=False), col] = np.nan
df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)

df.to_csv("data/student_data.csv", index=False)
print("Saved data/student_data.csv", df.shape)
