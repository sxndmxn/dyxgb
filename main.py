import polars as pl
from InquirerPy import inquirer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


def load_truth_data() -> pl.DataFrame:
    # mock your "truth" data from "database"
    num_samples = 50
    return pl.DataFrame(
        {
            "id": list(range(1, num_samples + 1)),
            "feature_1": [i * 0.5 for i in range(1, num_samples + 1)],
            "feature_2": [i % 7 for i in range(1, num_samples + 1)],
            "feature_3": [i % 3 for i in range(1, num_samples + 1)],
            "label": ["class_A" if i % 2 == 0 else "class_B" for i in range(1, num_samples + 1)],
        }
    )


def load_unknown_data() -> pl.DataFrame:
    # mock unknown_data from "database"
    num_samples = 20
    unknown_data_start_id = 1000
    return pl.DataFrame(
        {
            "id": list(range(unknown_data_start_id, unknown_data_start_id + num_samples)),
            "feature_1": [i * 0.45 for i in range(1, num_samples + 1)],
            "feature_2": [i % 7 for i in range(1, num_samples + 1)],
            "feature_3": [i % 3 for i in range(1, num_samples + 1)],
        }
    )


def select_target_and_features(df: pl.DataFrame):
    available_columns = df.columns
    feature_cols = inquirer.checkbox(
        message="Select feature columns for training (SPACE to toggle, ENTER to confirm):",
        choices=available_columns,
        validate=lambda res: len(res) > 0 or "Select at least one feature.",
    ).execute()

    remaining_columns = [column for column in available_columns if column not in feature_cols]
    target_col = inquirer.select(
        message="Select target label column:",
        choices=remaining_columns,
    ).execute()

    return target_col, list(feature_cols)


def align_unknown_data_columns(
    unknown_df: pl.DataFrame, selected_features: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    rename_map: dict[str, str] = {}
    for feature in selected_features:
        if feature not in unknown_df.columns:
            choices = list(unknown_df.columns) + ["<skip>"]
            mapped_column = inquirer.select(
                message=f"Unknown data is missing feature '{feature}'. Map an existing column to it or <skip>:",
                choices=choices,
            ).execute()
            if mapped_column != "<skip>":
                rename_map[mapped_column] = feature

    if rename_map:
        unknown_df = unknown_df.rename(rename_map)

    missing_features_after_mapping = [column for column in selected_features if column not in unknown_df.columns]
    if missing_features_after_mapping:
        print("Dropping features not present in unknown_data:", missing_features_after_mapping)
        selected_features = [column for column in selected_features if column in unknown_df.columns]

    return unknown_df, selected_features


def train_xgboost_classifier(truth_df: pl.DataFrame, target_col: str, feature_cols: list[str]):
    df_pd = truth_df.select(feature_cols + [target_col]).to_pandas()
    feature_matrix = df_pd[feature_cols]
    target_labels = df_pd[target_col]

    label_encoder = LabelEncoder()
    encoded_target_labels = label_encoder.fit_transform(target_labels)

    objective = "multi:softprob" if len(label_encoder.classes_) > 2 else "binary:logistic"

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        n_jobs=-1,
        random_state=42,
    )
    classifier.fit(feature_matrix, encoded_target_labels)
    return classifier, label_encoder


def save_artifacts(classifier: XGBClassifier, label_encoder: LabelEncoder):
    model_output_path = inquirer.text(
        message="Path to save XGBoost model (.json):",
        default="xgb_model.json",
    ).execute()
    label_encoder_output_path = inquirer.text(
        message="Path to save LabelEncoder (.joblib):",
        default="label_encoder.joblib",
    ).execute()

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(label_encoder_output_path).parent.mkdir(parents=True, exist_ok=True)

    classifier.save_model(model_output_path)
    joblib.dump(label_encoder, label_encoder_output_path)
    print(f"Saved model -> {model_output_path}")
    print(f"Saved label encoder -> {label_encoder_output_path}")


def predict_unknown_data_labels(
    classifier: XGBClassifier,
    label_encoder: LabelEncoder,
    unknown_df: pl.DataFrame,
    feature_cols: list[str],
):
    if not all(column in unknown_df.columns for column in feature_cols):
        missing_features = [column for column in feature_cols if column not in unknown_df.columns]
        print("Cannot predict, unknown_data missing features:", missing_features)
        return

    unk_pd = unknown_df.select(feature_cols).to_pandas()
    class_probabilities = classifier.predict_proba(unk_pd)
    predicted_class_indices = class_probabilities.argmax(axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_class_indices)
    prediction_confidences = class_probabilities.max(axis=1)

    predictions_df = unknown_df.with_columns(
        pl.Series("predicted_label", predicted_labels),
        pl.Series("confidence", prediction_confidences),
    )

    if inquirer.confirm(
        message="Save predictions for unknown_data?", default=True
    ).execute():
        predictions_output_path = inquirer.text(
            message="Path to save predictions (auto .parquet / .csv):",
            default="unknown_predictions.parquet",
        ).execute()
        Path(predictions_output_path).parent.mkdir(parents=True, exist_ok=True)
        if predictions_output_path.lower().endswith(".csv"):
            predictions_df.write_csv(predictions_output_path)
        else:
            predictions_df.write_parquet(predictions_output_path)
        print(f"Saved predictions -> {predictions_output_path}")


def main():
    print("Loading data from database...")
    truth_df = load_truth_data()
    unknown_df = load_unknown_data()

    target_col, feature_cols = select_target_and_features(truth_df)
    unknown_df, feature_cols = align_unknown_data_columns(unknown_df, feature_cols)

    print("Training XGBoost model...")
    classifier, label_encoder = train_xgboost_classifier(truth_df, target_col, feature_cols)
    save_artifacts(classifier, label_encoder)

    if inquirer.confirm(
        message="Run predictions on unknown_data now?", default=True
    ).execute():
        predict_unknown_data_labels(classifier, label_encoder, unknown_df, feature_cols)


if __name__ == "__main__":
    main()
