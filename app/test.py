import pandas as pd
import time

from agents          import OnwardJourneyAgent
from typing          import List, Dict, Any
from sklearn.metrics import confusion_matrix
from helpers         import extract_and_standardize_phone, get_encoded_labels_and_mapping
from loaders         import save_test_results
from plotting        import plot_uid_confusion_matrix
from metrics        import clarification_success_gain_metric
def prototype_one_test(oj_agent: OnwardJourneyAgent, test_queries : List[Dict[str, Any]], output_dir: str):
    """
    """
    return

def prototype_two_test(oj_agent: OnwardJourneyAgent, test_queries : List[Dict[str, Any]], output_dir: str):
    """
    """
    # clarification mode trial
    df_clarity = prototype_two_evaluation_trial(oj_agent, "CLARITY (Two-Turn)", test_queries, forced_mode=False)
    # one turn mode trial
    df_forced  = prototype_two_evaluation_trial(oj_agent, "FORCED (One-Turn)", test_queries, forced_mode=True)

    # Calculate Clarification Success Gain (CSG) Metric
    quantitative_metrics = clarification_success_gain_metric(df_clarity, df_forced)

    print("\n" + "=" * 80)
    print("Clarification Gain Analysis (AMBIGUOUS CASES ONLY)")
    print(f"Clarity (Success after 2 turns): {quantitative_metrics.get('ts_clarity_accuracy', 0):.2%}")
    print(f"Forced (Success when Forced):  {quantitative_metrics.get('ts_initial_accuracy', 0):.2%}")
    print(f"CLARIFICATION SUCCESS GAIN (CSG): {quantitative_metrics.get('clarification_success_gain_csg', 0):.4f}")
    print("\n" + "=" * 80)

    # Extract true and predicted labels for confusion matrix for clarification mode
    y_true_clarity = df_clarity['correct_phone'].tolist()
    y_pred_clarity = df_clarity['extracted_phone'].tolist()

    # get encoded labels and mapping
    y_true_encoded, y_pred_encoded, uid_labels, label_to_uid, all_codes = \
        get_encoded_labels_and_mapping(y_true_clarity, y_pred_clarity)

    # build confusion matrix dataframe
    cm_array = confusion_matrix(y_true_encoded, y_pred_encoded, labels=all_codes)
    cm_df    = pd.DataFrame(cm_array, index=uid_labels, columns=uid_labels)

    # get overall accuracy
    accuracy = (df_clarity['match_status'] == 'PASS').mean()

    # Plot and save confusion matrix and results
    fig               = plot_uid_confusion_matrix(cm_array, uid_labels, accuracy, title="Onward Journey Agent - Clarification Mode Confusion Matrix")
    save_test_results(cm_df=cm_df, output_dir=output_dir, fig=fig)
    return

def prototype_two_evaluation_trial(oj_agent: OnwardJourneyAgent, trial_name: str, test_cases: list, forced_mode: bool) -> pd.DataFrame:
    """
    Runs a single trial (clarification or forced) over the dataset.
    """
    print(f"\n--- Starting Trial: {trial_name} ---")
    results = []

    for i, test in enumerate(test_cases):

        # Reset agent history/state for clean run before each case
        oj_agent.history = []
        oj_agent.awaiting_clarification = False

        query = test['query']

        try:
            start_time = time.time()
            agent_response = ""

            if forced_mode:
                agent_response = oj_agent.get_forced_response(query)

            else: # clarification mode
                if test.get('is_ambiguous', False):
                    # Turn 1: Get clarification question
                    clarification_q = oj_agent._send_message_and_handle_tools(query)

                    if oj_agent.awaiting_clarification and "simulated_clarification_response" in test:
                        clarification_ans = test['simulated_clarification_response']
                        # Turn 2: Send clarification answer
                        agent_response = oj_agent._send_message_and_handle_tools(clarification_ans)
                    else:
                        agent_response = f"[FAIL] Agent did not ask question: {clarification_q[:50]}..."

                else: # Non-ambiguous cases run in a single turn
                    agent_response = oj_agent._send_message_and_handle_tools(query)

            end_time = time.time()

            extracted_phone = extract_and_standardize_phone(agent_response)
            match = 'PASS' if extracted_phone == test['expected_phone_number'] else 'FAIL'

            results.append({
                'test_id': test.get('test_id', i),
                'query': query,
                'correct_phone': test['expected_phone_number'],
                'extracted_phone': extracted_phone,
                'match_status': match,
                'is_ambiguous': test.get('is_ambiguous', False),
                'response_time_sec': round(end_time - start_time, 2)
            })
            print(f"Case {i+1} ({trial_name}): {match}. Extracted: {extracted_phone}. Expected: {test['expected_phone_number']}")

        except Exception as e:
            results.append({
                'test_id': test.get('test_id', i), 'query': query, 'correct_phone': test['expected_phone_number'],
                'extracted_phone': 'API_ERROR', 'match_status': 'ERROR', 'is_ambiguous': test.get('is_ambiguous', False),
                'response_time_sec': 0
            })
            print(f"Case {i+1} ({trial_name}): !!! ERROR during test: {e}")

    return pd.DataFrame(results)
