"""
Placement Readiness Predictor - training script
GDG-USAR Tech Team task (AI/ML domain)

Run:  python train.py
It cleans the data, creates the label, trains three models, prints
precision / recall / F1, explains one prediction and saves two charts.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files without needing a screen
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/placement_readiness.csv"

# Features the model is allowed to see.
NUMERIC = ["cgpa", "age"]
CATEGORICAL = ["backlogs", "course", "branch"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "ready"

# The readiness rule (see DECISIONS.md, decision 1).
CGPA_CUTOFF = 7.0
ALLOWED_BACKLOGS = ["0", "1"]


def load_and_clean(path=DATA_PATH):
    """Step 1-3: load the CSV, clean it, and create the label."""
    df = pd.read_csv(path)
    rows_raw = len(df)

    # 9,000 of the 10,000 rows are exact copies. If they stay, the same
    # student ends up in both train and test and the scores are inflated.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows: {rows_raw} raw -> {len(df)} after removing exact duplicates")

    # A blank 'backlogs' means the student has no backlogs, so it becomes
    # its own category "0" instead of being treated as unknown.
    df["backlogs"] = df["backlogs"].fillna("0").astype(str).str.strip()

    # Tidy text columns so "CSE " and "CSE" are not two different branches.
    for col in ["course", "branch"]:
        df[col] = df[col].astype(str).str.strip()

    # name, email  -> identifiers, tell us nothing about readiness
    # other_course -> filled only when course == "Other", 113 random values
    # gender       -> left out on purpose: it should not decide readiness
    # year         -> readiness here is about academics, not how far along
    #                 the student is (see DECISIONS.md, decision 6)
    df = df.drop(columns=["name", "email", "other_course", "gender", "year"])

    # The dataset has NO target column, so we define one with a clear rule.
    df[TARGET] = ((df["cgpa"] >= CGPA_CUTOFF)
                  & (df["backlogs"].isin(ALLOWED_BACKLOGS))).astype(int)
    return df


def make_pipeline(model):
    """Preprocessing + model in one object, so nothing leaks from test data."""
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", preprocess), ("model", model)])


def build_and_train():
    """Steps 4-7: split, then fit all three models on the training part only."""
    df = load_and_clean()
    X, y = df[FEATURES], df[TARGET]

    # Split BEFORE any scaling/encoding. stratify keeps the ready / not-ready
    # ratio the same in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Baseline (always majority)": make_pipeline(
            DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000)),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=4, random_state=42)),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return df, models, (X_train, X_test, y_train, y_test)


def main():
    df, models, (X_train, X_test, y_train, y_test) = build_and_train()
    print(f"Ready: {y_train.mean():.1%} of training students "
          f"(train={len(X_train)}, test={len(X_test)})\n")

    # ---- Step 8: compare models on the unseen test set -------------------
    rows = []
    for name, m in models.items():
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, m.predict(X_test), average="binary", zero_division=0)
        rows.append({"model": name, "precision": p, "recall": r, "f1": f1})
    print("Scores for the 'ready' class on the test set")
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    for name in ["Logistic Regression", "Decision Tree"]:
        print(f"--- {name} ---")
        print(classification_report(y_test, models[name].predict(X_test),
                                    target_names=["not ready", "ready"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ["Logistic Regression", "Decision Tree"]):
        ConfusionMatrixDisplay.from_estimator(
            models[name], X_test, y_test,
            display_labels=["not ready", "ready"], ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig("confusion_matrices.png", dpi=150)

    # ---- Step 9: one example prediction, explained -----------------------
    student = X_test.iloc[[0]]
    print("Example student:", student.iloc[0].to_dict())
    print("True label:", "ready" if y_test.iloc[0] else "not ready")
    for name in ["Logistic Regression", "Decision Tree"]:
        prob = models[name].predict_proba(student)[0, 1]
        print(f"  {name}: {prob:.1%} chance of being ready")

    # Why? Each feature's push = coefficient x the student's encoded value.
    lr = models["Logistic Regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    encoded = lr.named_steps["prep"].transform(student)
    encoded = encoded.toarray()[0] if hasattr(encoded, "toarray") else encoded[0]
    push = pd.Series(encoded * lr.named_steps["model"].coef_[0], index=names)
    push = push[push != 0].sort_values(key=abs, ascending=False)
    print("  Biggest pushes in Logistic Regression (+ towards ready):")
    print(push.head(4).round(2).to_string(), "\n")

    # ---- Step 10: which features matter? ---------------------------------
    # Shuffle one column at a time and see how much F1 drops.
    tree = models["Decision Tree"]
    imp = permutation_importance(tree, X_test, y_test, scoring="f1",
                                 n_repeats=20, random_state=42)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    print("Permutation importance (drop in F1 when the column is shuffled)")
    print(importance.round(3).sort_values(ascending=False).to_string())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    importance.plot.barh(ax=ax)
    ax.set_xlabel("Drop in F1 when shuffled")
    ax.set_title("Feature importance (Decision Tree)")
    fig.tight_layout()
    fig.savefig("feature_importance.png", dpi=150)
    print("\nSaved confusion_matrices.png and feature_importance.png")


if __name__ == "__main__":
    main()
