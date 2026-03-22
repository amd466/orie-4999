from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")


def evaluate_batch(batch_path):

    matched = pd.read_csv(batch_path / "matched_pairs.csv")
    unmatched = pd.read_csv(batch_path / "unmatched_mentees.csv")
    mentors_remaining = pd.read_csv(batch_path / "mentors_remaining_capacity.csv")
    comp_matrix = pd.read_csv(batch_path / "compatibility_matrix.csv")

    # -------------------------
    # Coverage
    # -------------------------

    n_matched = len(matched)
    n_unmatched = len(unmatched)
    coverage = n_matched / (n_matched + n_unmatched)

    # -------------------------
    # Compatibility distribution
    # -------------------------

    mean_score = matched["compatibility_score"].mean()
    median_score = matched["compatibility_score"].median()
    std_score = matched["compatibility_score"].std()

    # -------------------------
    # Top field match rate
    # -------------------------

    top_field_match_rate = (
        matched["mentee_top_field"] == matched["mentor_field"]
    ).mean()

    # -------------------------
    # Average field similarity
    # -------------------------

    avg_field_similarity = matched["field_similarity_score"].mean()

    # -------------------------
    # Mentor depletion
    # -------------------------

    remaining_capacity = mentors_remaining["remaining_capacity"].sum()

    # -------------------------
    # Opportunity inequality
    # -------------------------

    compatible = comp_matrix[comp_matrix["compatibility_score"] > 0]
    opp_counts = compatible.groupby("mentee_id")["mentor_id"].nunique()
    avg_opportunities = opp_counts.mean()

    # -------------------------
    # Mentor Load Balance      
    # -------------------------
    mentor_loads = matched["Mentor"].value_counts()
    load_mean = mentor_loads.mean()
    load_std = mentor_loads.std()
    load_cv = load_std / load_mean

    loads_sorted = sorted(mentor_loads.values)
    n = len(loads_sorted)
    gini = (
        2 * sum((i+1) * v for i, v in enumerate(loads_sorted))
    ) / (n * sum(loads_sorted)) - (n+1)/n

    # -------------------------
    # Score Percentile Distribution
    # -------------------------
    pct_above_05 = (matched["Compatibility_Score"] >= 0.50).mean()
    pct_above_06 = (matched["Compatibility_Score"] >= 0.60).mean()
    pct_above_07 = (matched["Compatibility_Score"] >= 0.70).mean()
    pct_above_08 = (matched["Compatibility_Score"] >= 0.80).mean()
    pct_above_09 = (matched["Compatibility_Score"] >= 0.90).mean()

    return {
        "batch": batch_path.name,
        "coverage": coverage,
        "mean_score": mean_score,
        "median_score": median_score,
        "std_score": std_score,
        "top_field_match_rate": top_field_match_rate,
        "avg_field_similarity": avg_field_similarity,
        "remaining_capacity": remaining_capacity,
        "avg_compatible_mentors": avg_opportunities,
        "load_cv": load_cv,
        "gini": gini,
        "pct_above_05": pct_above_05,
        "pct_above_06": pct_above_06,
        "pct_above_07": pct_above_07,
        "pct_above_08": pct_above_08,
        "pct_above_09": pct_above_09,
    }


def evaluate_all_batches():

    batch_folders = sorted(RESULTS_DIR.glob("batch_*"))

    results = []

    for batch in batch_folders:

        metrics = evaluate_batch(batch)

        results.append(metrics)

    df = pd.DataFrame(results)

    return df


def plot_metrics(df):

    plt.figure()
    plt.plot(df["batch"], df["remaining_capacity"], marker="o")
    plt.title("Mentor Depletion Curve")
    plt.ylabel("Remaining Mentor Capacity")
    plt.xlabel("Batch")
    plt.show()

    plt.figure()
    plt.plot(df["batch"], df["avg_compatible_mentors"], marker="o")
    plt.title("Opportunity Inequality")
    plt.ylabel("Avg Compatible Mentors")
    plt.xlabel("Batch")
    plt.show()

    plt.plot(df["batch"], df["gini"], marker="o")
    plt.title("Mentor Load Balance (Gini Coefficient)")
    plt.ylabel("Gini Coefficient")
    plt.xlabel("Batch")
    plt.show()

    plt.figure()
    plt.bar(
        ["≥0.5", "≥0.6", "≥0.7", "≥0.8", "≥0.9"],
        [df["pct_above_05"].mean(), df["pct_above_06"].mean(),
         df["pct_above_07"].mean(), df["pct_above_08"].mean(),
         df["pct_above_09"].mean()]
    )
    plt.title("Score Percentile Distribution")
    plt.ylabel("% of Matches")
    plt.xlabel("Compatibility Score Threshold")
    plt.show()


if __name__ == "__main__":

    summary = evaluate_all_batches()

    print(summary)

    summary.to_csv("algorithm_evaluation_summary.csv", index=False)

    plot_metrics(summary)
