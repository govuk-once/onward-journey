import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

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

def get_encoded_labels_and_mapping(
    y_true: List[str], y_pred: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, str], List[int]]:
    """Encodes string labels into integer codes and generates UID labels."""
    
    all_labels                    = pd.Series(y_true + y_pred).astype(str).unique()
    codes, unique_original_labels = pd.factorize(all_labels)
    
    label_to_code = {label: code for code, label in enumerate(unique_original_labels)}
    uid_labels    = [f"UID_{code}" for code in range(len(unique_original_labels))]
    label_to_uid  = {label: uid_labels[code] for label, code in label_to_code.items()}
    
    y_true_encoded = np.array([label_to_code[label] for label in y_true])
    y_pred_encoded = np.array([label_to_code.get(label, -1) for label in y_pred])
    return y_true_encoded, y_pred_encoded, uid_labels, label_to_uid, codes