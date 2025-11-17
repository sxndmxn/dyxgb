import polars as pl
from InquirerPy import inquirer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


def load_truth_data() -> pl.DataFrame:
    # mock your "truth" data from "database"
    n = 50
    return pl.DataFrame(
        {
            "id": list(range(1, n + 1)),
            "feature_1": [i * 0.5 for i in range(1, n + 1)],
            "feature_2": [i % 7 for i in range(1, n + 1)],
            "feature_3": [i % 3 for i in range(1, n + 1)],
            "label": ["class_A" if i % 2 == 0 else "class_B" for i in range(1, n + 1)],
        }
    )


def load_unknown_data() -> pl.DataFrame:
    # mock unknown_data from "database"
    n = 20
    start = 1000
    return pl.DataFrame(
        {
            "id": list(range(start, start + n)),
            "feature_1": [i * 0.45 for i in range(1, n + 1)],
            "feature_2": [i % 7 for i in range(1, n + 1)],
            "feature_3": [i % 3 for i in range(1, n + 1)],
        }
    )


def select_target_and_features(df: pl.DataFrame):
    cols = df.columns
    feature_cols = inquirer.checkbox(
        message="Select feature columns for training (SPACE to toggle, ENTER to confirm):",
        choices=cols,
        validate=lambda res: len(res) > 0 or "Select at least one feature.",
    ).execute()

    remaining = [c for c in cols if c not in feature_cols]
    target_col = inquirer.select(
        message="Select target label column:",
        choices=remaining,
    ).execute()

    return target_col, list(feature_cols)


def align_unknown_columns(
    unknown_df: pl.DataFrame, selected_features: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    rename_map: dict[str, str] = {}
    for feat in selected_features:
        if feat not in unknown_df.columns:
            choices = list(unknown_df.columns) + ["<skip>"]
            mapped = inquirer.select(
                message=f"Unknown data is missing feature '{feat}'. Map an existing column to it or <skip>:",
                choices=choices,
            ).execute()
            if mapped != "<skip>":
                rename_map[mapped] = feat

    if rename_map:
        unknown_df = unknown_df.rename(rename_map)

    missing_after = [c for c in selected_features if c not in unknown_df.columns]
    if missing_after:
        print("Dropping features not present in unknown_data:", missing_after)
        selected_features = [c for c in selected_features if c in unknown_df.columns]

    return unknown_df, selected_features


def train_xgb(truth_df: pl.DataFrame, target_col: str, feature_cols: list[str]):
    df_pd = truth_df.select(feature_cols + [target_col]).to_pandas()
    X = df_pd[feature_cols]
    y = df_pd[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    objective = "multi:softprob" if len(le.classes_) > 2 else "binary:logistic"

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y_enc)
    return clf, le


def save_artifacts(clf: XGBClassifier, le: LabelEncoder):
    model_path = inquirer.text(
        message="Path to save XGBoost model (.json):",
        default="xgb_model.json",
    ).execute()
    le_path = inquirer.text(
        message="Path to save LabelEncoder (.joblib):",
        default="label_encoder.joblib",
    ).execute()

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(le_path).parent.mkdir(parents=True, exist_ok=True)

    clf.save_model(model_path)
    joblib.dump(le, le_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved label encoder -> {le_path}")


def predict_unknowns(
    clf: XGBClassifier,
    le: LabelEncoder,
    unknown_df: pl.DataFrame,
    feature_cols: list[str],
):
    if not all(c in unknown_df.columns for c in feature_cols):
        missing = [c for c in feature_cols if c not in unknown_df.columns]
        print("Cannot predict, unknown_data missing features:", missing)
        return

    unk_pd = unknown_df.select(feature_cols).to_pandas()
    proba = clf.predict_proba(unk_pd)
    preds_idx = proba.argmax(axis=1)
    preds = le.inverse_transform(preds_idx)
    confidences = proba.max(axis=1)

    out_df = unknown_df.with_columns(
        pl.Series("predicted_label", preds),
        pl.Series("confidence", confidences),
    )

    if inquirer.confirm(
        message="Save predictions for unknown_data?", default=True
    ).execute():
        out_path = inquirer.text(
            message="Path to save predictions (auto .parquet / .csv):",
            default="unknown_predictions.parquet",
        ).execute()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.lower().endswith(".csv"):
            out_df.write_csv(out_path)
        else:
            out_df.write_parquet(out_path)
        print(f"Saved predictions -> {out_path}")


def main():
    print("Loading data from database...")
    truth_df = load_truth_data()
    unknown_df = load_unknown_data()

    target_col, feature_cols = select_target_and_features(truth_df)
    unknown_df, feature_cols = align_unknown_columns(unknown_df, feature_cols)

    print("Training XGBoost model...")
    clf, le = train_xgb(truth_df, target_col, feature_cols)
    save_artifacts(clf, le)

    if inquirer.confirm(
        message="Run predictions on unknown_data now?", default=True
    ).execute():
        predict_unknowns(clf, le, unknown_df, feature_cols)


if __name__ == "__main__":
    main()
