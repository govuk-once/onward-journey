import pandas as pd
import re
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from agents import OnwardJourneyAgent

# --- Helper Functions ---

def clean_phone_number_for_matrix(text: str) -> str:
    """
    Tries to extract a UK phone number and standardizes the format
    (e.g., '0300 200 3887') to match the expected class labels in the CSV.

    The function is adjusted to match common 4-3-3 and 4-2-2-2 segmented formats.
    """

    # Pattern 1: Common non-geographic/mobile-like split (e.g., 4-3-3)
    pattern_4_3_X = r'\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4}'

    # Pattern 2: Common freephone/geographic split (e.g., 4-2-2-2)
    # This covers the 0800 80 70 60 example
    pattern_4_2_2_2 = r'\d{4}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}'

    # Combine the patterns using alternation (|) and ensure boundary matching
    combined_pattern = r'\b(' + pattern_4_3_X + r'|' + pattern_4_2_2_2 + r')\b'

    match = re.search(combined_pattern, text)
    if match:
        # 1. Clean up: remove spaces and hyphens
        # match.group(1) is the entire number string matched by either pattern
        extracted_num_cleaned = match.group(1).replace(' ', '').replace('-', '')

        # 2. Re-format to the standard output format (4-3-X for consistency)
        # This standardizes both input formats (4-3-3 and 4-2-2-2) to one output format (e.g., 0800 807 060).
        if len(extracted_num_cleaned) >= 10:
             return extracted_num_cleaned[0:4] + ' ' + extracted_num_cleaned[4:7] + ' ' + extracted_num_cleaned[7:]

        # Fallback for shorter numbers
        return ' '.join(extracted_num_cleaned[i:i+3] for i in range(0, len(extracted_num_cleaned), 3)).strip()

    return 'NOT FOUND' # The class label for misclassification

def load_test_queries(file_path: str) -> list[dict]:
    """
    Loads test queries from the specified CSV file.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: Test queries file not found at {file_path}")
        return []

    df = pd.read_csv(file_path)

    test_queries = []
    for _, row in df.iterrows():
        # Ensure the correct phone number is standardized when loading
        standardized_phone = clean_phone_number_for_matrix(str(row['phone_number']))

        test_queries.append({
            'uid': row['uid'],
            'correct_phone': standardized_phone,
            'query': row['Question']
        })

    return test_queries

def test_agent_with_queries(oj_agent: OnwardJourneyAgent, test_queries: list):
    """
    Runs all predefined test queries through the Onward Journey Agent,
    validates the extracted phone number against the expected number,
    and saves a summary report to CSV.
    """
    if not test_queries:
        print("No test queries loaded. Skipping test.")
        return

    print("----------------------------------------------------------------------------------------------------")
    print("STARTING MASS TEST: VERIFYING PHONE NUMBER RETRIEVAL")
    print(f"Total queries to test: {len(test_queries)}")
    print("NOTE: Inserting a 6-second delay between queries to respect API rate limits.")
    print("----------------------------------------------------------------------------------------------------")

    results = []
    pass_count = 0

    for i, test in enumerate(test_queries):

        print(f"Testing Query {i+1}/{len(test_queries)} ({test['uid']}): {test['query']}")

        correct_phone = test['correct_phone']

        try:
            start_time = time.time()
            agent_response = oj_agent.get_final_response(test['query'])
            end_time = time.time()

            extracted_phone = clean_phone_number_for_matrix(agent_response)

            match = 'FAIL'
            if extracted_phone == correct_phone:
                match = 'PASS'
                pass_count += 1

            results.append({
                'uid': test['uid'],
                'query': test['query'],
                'correct_phone': correct_phone,
                'agent_response_snippet': agent_response[:100].replace('\n', ' ') + '...' if len(agent_response) > 100 else agent_response.replace('\n', ' '),
                'extracted_phone': extracted_phone,
                'match_status': match,
                'response_time_sec': round(end_time - start_time, 2)
            })
            print(f"-> Status: {match}. Extracted: {extracted_phone}. Expected: {correct_phone}")

        except Exception as e:
            results.append({
                'uid': test['uid'],
                'query': test['query'],
                'correct_phone': correct_phone,
                'agent_response_snippet': f"ERROR: {e}",
                'extracted_phone': 'API_ERROR',
                'match_status': 'ERROR',
                'response_time_sec': 0
            })
            print(f"-> ERROR during test: {e}")

            if "RESOURCE_EXHAUSTED" in str(e):
                print("!!! RESOURCE_EXHAUSTED detected. Pausing for 90 seconds for better recovery. !!!")
                time.sleep(90)

        # 3. Mandatorily wait 10 seconds before the next query to avoid hitting the rate limit
        if i < len(test_queries) - 1:
            # INCREASED WAIT TIME to ensure compliance with 10 requests/minute limit
            time.sleep(10)

    # --- Generate Report ---
    report_df = pd.DataFrame(results)
    report_file = 'onward_journey_test_report.csv'
    report_df.to_csv(report_file, index=False)

    print("-" * 50)
    print("MASS TEST COMPLETE")
    print(f"Total Queries: {len(test_queries)}")
    print(f"Total Matches (PASS): {pass_count}")
    print(f"Total Failures (FAIL/ERROR): {len(test_queries) - pass_count}")
    print(f"Detailed report saved to {report_file}")
    print("-" * 50)

    # Generate the confusion matrix summary and plot
    generate_multi_class_confusion_matrix(report_file, output_dir='/Users/alexhiles/Desktop/onwardjourneytool/test_output')

    return report_file

def generate_multi_class_confusion_matrix(file_path: str, output_dir: str = 'test_output'):
    """
    Loads the test report and generates a multi-class confusion matrix,
    printing the text table and saving a Matplotlib heatmap.
    """
    df = pd.read_csv(file_path)

    df['Actual_Phone'] = df['correct_phone']
    df['Predicted_Phone'] = df['extracted_phone']

    # Create the confusion matrix table
    confusion_matrix = pd.crosstab(
        df['Actual_Phone'],
        df['Predicted_Phone'],
        rownames=['Actual Phone Number'],
        colnames=['Predicted Phone Number'],
        dropna=False
    )

    # Calculate the overall accuracy for context
    total_predictions = len(df)
    correct_predictions = sum(df['Actual_Phone'] == df['Predicted_Phone'])
    overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    # --- Text Output ---
    print("\n" + "=" * 80)
    print(f"MULTI-CLASS CONFUSION MATRIX REPORT (Total Queries: {total_predictions})")
    print("=" * 80)
    print(confusion_matrix.to_string())
    print(f"\nOVERALL QUERY-TO-CONTACT ACCURACY: {overall_accuracy:.2f}%")
    print("=" * 80)

    # --- Matplotlib/Seaborn Plot ---

    # 1. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, 'confusion_matrix_heatmap.png')

    plt.figure(figsize=(16, 12))

    # Use log normalization if counts vary widely, otherwise use standard.
    # Since counts are 1-10, standard is fine. Use annot=True for cell values.
    # Set display format for integers ('d')
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Number of Queries'}
    )

    plt.title(f'Agent Misclassification Heatmap (Accuracy: {overall_accuracy:.2f}%)', fontsize=18)
    plt.xlabel('Predicted Phone Number', fontsize=14)
    plt.ylabel('Actual Phone Number', fontsize=14)

    # Adjust layout to prevent truncation
    plt.tight_layout()

    # Save the figure to the specified output directory
    plt.savefig(plot_file)
    plt.close() # Close the figure to free memory

    print(f"\n[Visual Report Saved] Matplotlib heatmap saved to: {plot_file}")

# The main execution function to be called from main.py
def run_mass_tests(oj_agent: OnwardJourneyAgent, test_data_path: str = "user_prompts.csv"):
    """
    Entry point to run the entire testing and analysis pipeline.
    """
    test_queries = load_test_queries(test_data_path)
    test_agent_with_queries(oj_agent, test_queries)
