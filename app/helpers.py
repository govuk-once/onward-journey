import re

from typing import Optional
from pydantic                 import BaseModel

class SearchResult(BaseModel):
    url: str
    score: float
    document_type: str
    title: str
    description: Optional[str]
    heading_hierarchy: list[str]
    html_content: str

def extract_and_standardize_phone(text: str) -> str:
    """
    Tries to extract a UK phone number and standardizes the format
    (e.g., '0300 200 3887') to match the expected class labels.
    """

    # Pattern 1: Common non-geographic/mobile-like split (e.g., 4-3-X)
    pattern_4_3_X = r'\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4}'

    # Pattern 2: Common freephone/geographic split (e.g., 4-2-2-2)
    pattern_4_2_2_2 = r'\d{4}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}'

    # Combine the patterns
    combined_pattern = r'\b(' + pattern_4_3_X + r'|' + pattern_4_2_2_2 + r')\b'

    match = re.search(combined_pattern, text)
    if match:
        # 1. Clean up: remove spaces and hyphens
        extracted_num_cleaned = match.group(1).replace(' ', '').replace('-', '')

        # 2. Re-format to the standard output format (4-3-X for consistency)
        if len(extracted_num_cleaned) >= 10:
             return extracted_num_cleaned[0:4] + ' ' + extracted_num_cleaned[4:7] + ' ' + extracted_num_cleaned[7:]

        return ' '.join(extracted_num_cleaned[i:i+3] for i in range(0, len(extracted_num_cleaned), 3)).strip()

    return 'NOT_FOUND' # Consistent misclassification label

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
