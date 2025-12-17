import os
from typing import Any, Dict, List
import json
import pandas as pd
import matplotlib.pyplot as plt

from helpers import extract_and_standardize_phone

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
    output_dir   : str = "test_output/prototype2"
) -> None:
    """Saves the test artifacts to the fixed output directory."""

    os.makedirs(output_dir, exist_ok=True)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix_uid.csv"))

    plot_file_path = os.path.join(output_dir, "confusion_matrix_plot.png")
    fig.savefig(plot_file_path)

    plt.close(fig)

    return
