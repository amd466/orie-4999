from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")


def evaluate_batch(batch_path):
    matched = pd.read_csv(batch_path / "matched_pairs.csv")
    unmatched = pd.read_csv(batch_path / "unmatched_mentees.csv")
    mentors_remaining = pd.read_csv(batch_path / "mentors_remaining_capacity.csv")
    comp_matrix = pd.read_csv(batch_path / "compatibility_matrix.csv")

    # Coverage
    n_matched = len(matched)
    n_unmatched = len(unmatched)
    coverage = n_matched / (n_matched + n_unmatched)

    # Compatibility score stats
    mean_score = matched["Compatibility_Score"].mean()
    median_score = matched["Compatibility_Score"].median()
    std_score = matched["Compatibility_Score"].std()

    # Mentor depletion
    remaining_capacity = mentors_remaining["Remaining_Capacity"].sum()

    # Opportunity inequality
    compatible = comp_matrix[comp_matrix["Compatibility_Score"] > 0]
    opp_counts = compatible.groupby("mentee_id")["mentor_id"].nunique()
    avg_opportunities = opp_counts.mean()

    # Mentor load balance
    mentor_loads = matched["Mentor"].value_counts()
    load_mean = mentor_loads.mean()
    load_std = mentor_loads.std()
    load_cv = load_std / load_mean  #coefficient of variation; lower = more balanced

    loads_sorted = sorted(mentor_loads.values)
    n = len(loads_sorted)
    gini = (
        2 * sum((i + 1) * v for i, v in enumerate(loads_sorted))
    ) / (n * sum(loads_sorted)) - (n + 1) / n
    #Gini coefficient: 0 = perfectly equal load, 1 = fully concentrated

    # Score percentile distribution
    pct_above_05 = (matched["Compatibility_Score"] >= 0.50).mean()
    pct_above_06 = (matched["Compatibility_Score"] >= 0.60).mean()
    pct_above_07 = (matched["Compatibility_Score"] >= 0.70).mean()
    pct_above_08 = (matched["Compatibility_Score"] >= 0.80).mean()
    pct_above_09 = (matched["Compatibility_Score"] >= 0.90).mean()

    # -------------------------
    # Remaining capacity by Mentor_Field_1
    # Each mentor's remaining slots are counted only under their top field.
    # Field lookup is built from matched_pairs.csv which contains Mentor_Field_1.
    # Note: mentors with remaining capacity but zero matches will not appear in
    # matched and will be unmapped (NaN field) — they are excluded from this metric.
    # -------------------------
    field_lookup = (
        matched[["Mentor", "Mentor_Field_1"]]
        .drop_duplicates(subset="Mentor")
        .set_index("Mentor")["Mentor_Field_1"]
    )
    mentors_remaining["Field"] = mentors_remaining["Mentor"].map(field_lookup)
    remaining_by_field = (
        mentors_remaining
        .dropna(subset=["Field"])
        .groupby("Field")["Remaining_Capacity"]
        .sum()
        .add_prefix("remaining_capacity_field_")
        .to_dict()
    )

    return {
        "batch": batch_path.name,
        # Coverage
        "coverage": coverage,
        # Score stats
        "mean_score": mean_score,
        "median_score": median_score,
        "std_score": std_score,
        # Mentor depletion (total)
        "remaining_capacity": remaining_capacity,
        # Opportunity inequality
        "avg_compatible_mentors": avg_opportunities,
        # Mentor load balance
        "load_mean": load_mean,
        "load_std": load_std,
        "load_cv": load_cv,
        "load_gini": gini,
        # Score percentile distribution
        "pct_above_0.50": pct_above_05,
        "pct_above_0.60": pct_above_06,
        "pct_above_0.70": pct_above_07,
        "pct_above_0.80": pct_above_08,
        "pct_above_0.90": pct_above_09,
        # Remaining capacity by top field (Mentor_Field_1)
        **remaining_by_field,
    }


def evaluate_all_batches():
    batch_folders = sorted(RESULTS_DIR.glob("batch_*"))
    results = [evaluate_batch(batch) for batch in batch_folders]
    return pd.DataFrame(results)


def plot_metrics(df):
    # --- Mentor depletion curve ---
    plt.figure()
    plt.plot(df["batch"], df["remaining_capacity"], marker="o")
    plt.title("Mentor depletion curve")
    plt.ylabel("Remaining mentor capacity")
    plt.xlabel("Batch")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Opportunity inequality ---
    plt.figure()
    plt.plot(df["batch"], df["avg_compatible_mentors"], marker="o")
    plt.title("Opportunity inequality")
    plt.ylabel("Avg compatible mentors per mentee")
    plt.xlabel("Batch")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Mentor load balance (CV and Gini) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["batch"], df["load_cv"], marker="o", color="steelblue")
    axes[0].set_title("Mentor load — coefficient of variation")
    axes[0].set_ylabel("CV (lower = more balanced)")
    axes[0].set_xlabel("Batch")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(df["batch"], df["load_gini"], marker="o", color="coral")
    axes[1].set_title("Mentor load — Gini coefficient")
    axes[1].set_ylabel("Gini (0 = equal, 1 = concentrated)")
    axes[1].set_xlabel("Batch")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Remaining capacity by Mentor_Field_1 ---
    field_cols = [c for c in df.columns if c.startswith("remaining_capacity_field_")]
    if field_cols:
        field_labels = [c.replace("remaining_capacity_field_", "") for c in field_cols]
        plt.figure(figsize=(10, 5))
        for col, label in zip(field_cols, field_labels):
            plt.plot(df["batch"], df[col].fillna(0), marker="o", label=label)
        plt.title("Remaining mentor capacity by field (Mentor_Field_1)")
        plt.ylabel("Total remaining capacity")
        plt.xlabel("Batch")
        plt.legend(title="Field", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # --- Score percentile distribution ---
    pct_cols = [
        "pct_above_0.50",
        "pct_above_0.60",
        "pct_above_0.70",
        "pct_above_0.80",
        "pct_above_0.90",
    ]
    plt.figure(figsize=(10, 5))
    for col in pct_cols:
        plt.plot(df["batch"], df[col], marker="o", label=col.replace("pct_above_", "≥ "))
    plt.title("Score percentile distribution across batches")
    plt.ylabel("Fraction of matched pairs")
    plt.xlabel("Batch")
    plt.legend(title="Compatibility threshold", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    summary = evaluate_all_batches()
    print(summary.to_string(index=False))
    summary.to_csv("algorithm_evaluation_summary.csv", index=False)
    plot_metrics(summary)
