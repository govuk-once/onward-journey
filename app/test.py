import pandas as pd
import time

from agents          import OnwardJourneyAgent
from typing          import List, Dict, Any
from sklearn.metrics import confusion_matrix
from helpers         import extract_and_standardize_phone, get_encoded_labels_and_mapping
from loaders         import save_test_results
from plotting        import plot_uid_confusion_matrix

class Evaluator:
    def __init__(self, oj_agent: OnwardJourneyAgent, test_queries: List[Dict[str, Any]], output_dir: str):

        self.oj_agent     = oj_agent
        self.test_queries = test_queries
        self.output_dir   = output_dir

        self.phone_to_topic = self._build_phone_to_topic_mapping()

    def __call__(self, run_mode: str) -> pd.DataFrame:
        results = self.forced_mode_evaluation() if run_mode == 'forced' else self.clarification_mode_evaluation()
        self.save_results(results, run_mode)
        self.plot_and_save_cm(results, run_mode)
        return results

    def _build_phone_to_topic_mapping(self) -> Dict[str, str]:
        """
        Constructs a mapping from phone numbers to their corresponding topics.
        """
        phone_to_topic = {}
        for test in self.test_queries:
            phone = str(test.get('expected_phone_number'))
            topic = test.get('topic', 'N/A')
            phone_to_topic[phone] = topic
        phone_to_topic["UNKNOWN"] = "UNKNOWN / FAILURE"

        return phone_to_topic

    def _oj_agent_eval(self, query: str) -> str:
        """
        Sends a single query to the OnwardJourneyAgent and returns the response.
        """
        response = self.oj_agent._send_message_and_tools(query)
        return response
    def _queries_eval(self, forced: bool) -> pd.DataFrame:
        """
        Evaluates all test queries and returns the list of agent responses.
        """
        results = []
        for idx, test in enumerate(self.test_queries):
            self.oj_agent.history = []  # Reset history for each test case
            query                 = test['query']
            try:
                start_time = time.time()

                forced_prompt  = ""
                agent_response = ""

                if forced:
                    # Force the agent to skip the 'Clarification Check' instruction
                    # We do this by passing a directive in the immediate user prompt
                    forced_prompt = f"INSTRUCTION: Ignore ambiguity. Use your tools immediately to answer: {query}"
                    agent_response = self._oj_agent_eval(forced_prompt)
                else:
                    # Turn 1: Initial Query
                    agent_response = self._oj_agent_eval(query)
                    # Turn 2: Check if simulated response is needed for ambiguous cases
                    # Note: oj_agent doesn't have 'awaiting_clarification' attribute,
                    # so we check if the response looks like a question or lacks tool output.
                    if test.get('is_ambiguous', False) and "simulated_clarification_response" in test:
                        clarification_ans = test['simulated_clarification_response']
                        # Follow up with the clarification answer
                        agent_response = self._oj_agent_eval(clarification_ans)

                end_time        = time.time()
                extracted_phone = extract_and_standardize_phone(agent_response)
                match           = 'PASS' if extracted_phone == test['expected_phone_number'] else 'FAIL'

                results.append({
                    'test_id': test.get('test_id', idx),
                    'query': query,
                    'correct_phone': test['expected_phone_number'],
                    'extracted_phone': extracted_phone,
                    'match_status': match,
                    'is_ambiguous': test.get('is_ambiguous', False),
                    'response_time_sec': round(end_time - start_time, 2),
                    'topic': test.get('topic', 'N/A'),
                    'conversation': self.oj_agent.history
                })
                print(f"Case {idx+1}: {match}. Expected: {test['expected_phone_number']}, Got: {extracted_phone}")

            except Exception as e:
                print(f"Case {idx+1}: !!! ERROR: {e}")
                results.append({
                    'test_id': test.get('test_id', idx), 'query': query, 'match_status': 'ERROR'
                })
        return pd.DataFrame(results)

    def forced_mode_evaluation(self) -> pd.DataFrame:
        """
        Evaluates the agent in forced mode (one-turn).
        """
        return self._queries_eval(forced=True)
    def clarification_mode_evaluation(self) -> pd.DataFrame:
        """
        Evaluates the agent in clarification mode (two-turn).
        """
        return self._queries_eval(forced=False)

    def plot_and_save_cm(self, df: pd.DataFrame, mode: str):
        """
        Groups results by Topic and generates the Confusion Matrix.
        """
        # 1. Ground Truth: Use the 'topic' column directly from the dataframe
        print(df)
        y_true = df['topic'].fillna("UNKNOWN / FAILURE").tolist()

        # 2. Prediction: Map the agent's extracted phone back to a topic
        # To handle the many-to-one issue, we use a mapping that checks if the
        # extracted phone matches the EXPECTED phone for that specific test case.
        y_pred = []
        for _, row in df.iterrows():
            extracted = str(row['extracted_phone'])
            #expected = str(row['correct_phone'])

            if row['match_status'] == 'PASS':
                # If the agent got the number right, the predicted topic IS the ground truth topic
                y_pred.append(row['topic'])
            elif extracted in [None, 'API_ERROR', '', 'None', 'unknown', 'nan', 'NOT_FOUND']:
                y_pred.append("UNKNOWN / FAILURE")
            else:
                # If it failed but got A number, try to find what topic that number belongs to
                # (Fallback to the first topic found for that number in your map)
                y_pred.append(self.phone_to_topic.get(extracted, "UNKNOWN / FAILURE"))

        # 3. Create a unique list of all topics from the original dataset
        all_possible_topics = sorted(list(set([t['topic'] for t in self.test_queries] + ["UNKNOWN / FAILURE"])))

        # 4. Use helper to get encoded values
        y_true_enc, y_pred_enc, labels, mapping, codes = get_encoded_labels_and_mapping(
            y_true,
            y_pred,
            custom_all_labels=all_possible_topics
        )

            # 5. Generate and Save Confusion Matrix
        cm_array = confusion_matrix(y_true_enc, y_pred_enc, labels=codes)
        cm_df    = pd.DataFrame(cm_array, index=labels, columns=labels)

        accuracy = (df['match_status'] == 'PASS').mean()
        fig = plot_uid_confusion_matrix(
            cm_array,
            labels,
            accuracy,
            title=f"Onward Journey Agent: {mode.capitalize()} Mode\n(Labels: Topic)"
        )

        # Save the figure and dataframe
        save_test_results(cm_df=cm_df, output_dir=self.output_dir, fig=fig, file_suffix=mode)
        print(f"Confusion Matrix saved for {mode} mode grouped by Topic.")
    def save_results(self, df: pd.DataFrame, mode: str):
        """
        Saves the evaluation results to a CSV file.
        """
        df.to_csv(f"{self.output_dir}/evaluation_results_{mode.lower()}.csv", index=False)
        print(f"Saved evaluation results for {mode} mode to CSV.")
