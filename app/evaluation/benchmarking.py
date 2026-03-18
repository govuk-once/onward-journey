import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Dict, List

from app.core.data import extract_and_standardize_phone

def load_test_queries(file_path: str) -> list[dict]:
    """
    Loads test queries from the specified file path, supporting both
    structured JSON (for multi-turn tests) and CSV (for simple tests).
    """
    if not os.path.exists(file_path):
        print(f"ERROR: Test queries file not found at {file_path}")
        return []

    file_extension = os.path.splitext(file_path)[1].lower()
    test_cases: List[Dict[str, Any]] = []

    if file_extension == '.json':
        try:
            with open(file_path, 'r') as f:
                test_cases = json.load(f)
            print(f"Loaded {len(test_cases)} cases from JSON file.")
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON from {file_path}. Details: {e}")
            return []

    elif file_extension == '.csv':
        try:
            df = pd.read_csv(file_path)
            required_cols = {'uid', 'Question', 'phone_number'}
            if not required_cols.issubset(df.columns):
                print(f"ERROR: CSV file must contain columns: {required_cols}.")
                return []

            for _, row in df.iterrows():
                test_cases.append({
                    'test_id': str(row['uid']),
                    'query': str(row['Question']),
                    'expected_phone_number': str(row['phone_number']),
                    'is_ambiguous': False,
                    'simulated_clarification_response': 'N/A'
                })
            print(f"Loaded {len(test_cases)} cases from CSV file (set to one-turn mode).")
        except Exception as e:
            print(f"ERROR: Failed to read or process CSV file. Details: {e}")
            return []

    else:
        print(f"ERROR: Unsupported file format: {file_extension}. Must be .csv or .json.")
        return []

    # Final Standardization Step (Applied to all loaded cases)
    for case in test_cases:
        case['expected_phone_number'] = extract_and_standardize_phone(case['expected_phone_number'])

    return test_cases

def save_test_results(
    cm_df        : pd.DataFrame,
    fig          : plt.Figure,
    output_dir   : str = "test_output/prototype2",
    file_suffix  : str = "",
) -> None:
    """Saves the test artifacts to the fixed output directory."""

    os.makedirs(output_dir, exist_ok=True)
    cm_df.to_csv(os.path.join(output_dir, f"confusion_matrix_uid{file_suffix}.csv"))

    plot_file_path = os.path.join(output_dir, f"confusion_matrix_plot{file_suffix}.png")
    fig.savefig(plot_file_path)

    plt.close(fig)

    return

def clarification_success_gain_metric(df_clarity: pd.DataFrame, df_forced: pd.DataFrame) -> dict:

    """Calculates the Clarification Success Gain (CSG)."""

    df_clarity_amb = df_clarity[df_clarity['is_ambiguous'] == True].copy()
    df_forced_amb  = df_forced[df_forced['is_ambiguous'] == True].copy()

    if df_clarity_amb.empty or df_forced_amb.empty: return {}

    df_clarity_amb = df_clarity_amb.set_index('test_id')
    df_forced_amb  = df_forced_amb.set_index('test_id')

    ts_clarity = (df_clarity_amb['match_status'] == 'PASS').mean()
    ts_initial = (df_forced_amb['match_status'] == 'PASS').mean()

    COST_PENALTY_C = 0.0 # Placeholder for cost penalty
    csg            = (ts_clarity - ts_initial) - COST_PENALTY_C

    metrics = {
        'total_ambiguous_cases': len(df_clarity_amb),
        'ts_clarity_accuracy': round(ts_clarity, 4),
        'ts_initial_accuracy': round(ts_initial, 4),
        'clarification_success_gain_csg': round(csg, 4),
    }
    return metrics

def plot_uid_confusion_matrix(cm_array, labels, accuracy, title="Confusion Matrix"):
    """
    Plots a confusion matrix using Service Names and ensures labels are formatted correctly.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a heatmap with integer formatting for counts
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, cbar=False)

    ax.set_title(f"{title}\nOverall Accuracy: {accuracy:.2%}", pad=20)
    ax.set_xlabel('Predicted Service', labelpad=15)
    ax.set_ylabel('True Service', labelpad=15)

    # Rotate tick labels to prevent "eating" into the plot
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0)

    # Adjust layout to make room for long labels
    plt.tight_layout()

    return fig

def get_encoded_labels_and_mapping(y_true, y_pred, custom_all_labels=None, semantic_mapping=None):
    """
    Standardizes label encoding for the confusion matrix.

    Args:
        y_true (list): Standardized ground truth labels.
        y_pred (list): Standardized predicted labels.
        custom_all_labels (list): The global set of labels (including UNKNOWN) for consistent axes.
        semantic_mapping (dict): Optional mapping from phone/UID to a friendly name.
    """
    # 1. Use the provided global set, or derive from input if not provided
    if custom_all_labels is None:
        all_unique_labels = sorted(list(set(y_true + y_pred)))
    else:
        all_unique_labels = sorted(custom_all_labels)

    # 2. Create the integer mapping (codes) for sklearn's confusion_matrix
    label_to_code = {label: i for i, label in enumerate(all_unique_labels)}

    # 3. Encode the input lists
    y_true_encoded = [label_to_code[label] for label in y_true]
    y_pred_encoded = [label_to_code[label] for label in y_pred]

    # 4. Create semantic display labels for the chart axes
    if semantic_mapping:
        # Map the phone/UID to a name, defaulting to the value itself if not found
        display_labels = [semantic_mapping.get(label, label) for label in all_unique_labels]
    else:
        display_labels = all_unique_labels

    # Return everything needed for the confusion matrix and plotting
    return y_true_encoded, y_pred_encoded, display_labels, label_to_code, list(label_to_code.values())
