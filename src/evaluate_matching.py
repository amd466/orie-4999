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


if __name__ == "__main__":

    summary = evaluate_all_batches()

    print(summary)

    summary.to_csv("algorithm_evaluation_summary.csv", index=False)

    plot_metrics(summary)