import random
import numpy as np
import argparse
import os  # Required for directory management

from data                        import vectorStore
from agents                      import OnwardJourneyAgent
from loaders                     import load_test_queries
from metrics                     import clarification_success_gain_metric

from test import Evaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KB_PATH = os.path.join(SCRIPT_DIR, "../mock_data/mock_rag_data.csv")

def default_handoff():
    return {'handoff_agent_id': 'GOV.UK Chat', 'final_conversation_history': []}

class AgentRunner:
    def __init__(self, llm_model_id: str, path_to_kb: str, path_to_test_data: str,
                 aws_region: str, aws_role_arn : str, output_dir: str,
                 seed: int = 0, vector_store_model_id: str = 'amazon.titan-embed-text-v2:0'):

        self.model_id                  = llm_model_id
        self.vector_store_model_id     = vector_store_model_id
        self.path_to_knowledge_base    = path_to_kb
        self.aws_region                = aws_region
        self.aws_role_arn              = aws_role_arn
        self.path_to_test_data         = path_to_test_data
        self.seed                      = seed
        self.output_dir                = output_dir

        self._set_all_seeds(self.seed)

    def __call__(self, run_mode: str, handoff_data: dict, top_k_oj: int, top_k_govuk: int):
        """
        Executes the agent with specific Top-K weightings and saves results
        to a unique sub-folder.
        """
        vs = self._initialize_vector_store()

        # Initialize agent with the specific K-pair
        oj_agent = self._initialize_agent(
            vs=vs,
            handoff_data=handoff_data,
            temperature=0.0,
            top_k_oj=top_k_oj,
            top_k_govuk=top_k_govuk
        )

        if run_mode == 'test':
            # Create a unique folder for this specific pair
            pair_folder_name = f"oj{top_k_oj}_gov{top_k_govuk}"
            current_run_dir = os.path.join(self.output_dir, pair_folder_name)
            os.makedirs(current_run_dir, exist_ok=True)

            print(f"Executing Test Suite. Results will be saved to: {current_run_dir}")
            test_queries = load_test_queries(self.path_to_test_data)

            # Pass the new sub-folder to the Evaluator
            evaluator = Evaluator(oj_agent, test_queries, current_run_dir)

            print(f"Running Forced Mode (K_oj={top_k_oj}, K_gov={top_k_govuk})...")
            forced_df = evaluator('forced')

            print(f"Running Clarification Mode (K_oj={top_k_oj}, K_gov={top_k_govuk})...")
            clarification_df = evaluator('clarification')

            gain_metrics = clarification_success_gain_metric(clarification_df, forced_df)

            print(f"CSG Score for {pair_folder_name}: {gain_metrics.get('clarification_success_gain_csg', 0):.4f}")

        elif run_mode == 'interactive':
            oj_agent.run_conversation()

    def _set_all_seeds(self, seed_value: int):
        random.seed(seed_value)
        np.random.seed(seed_value)

    def _initialize_vector_store(self):
        return vectorStore(file_path=self.path_to_knowledge_base)

    def _initialize_agent(self, vs: vectorStore, handoff_data: dict, temperature: float,
                          top_k_oj: int, top_k_govuk: int) -> OnwardJourneyAgent:
        return OnwardJourneyAgent(
                   handoff_package=handoff_data,
                   vector_store_embeddings=vs.get_embeddings(),
                   vector_store_chunks=vs.get_chunks(),
                   embedding_model=self.vector_store_model_id,
                   model_name=self.model_id,
                   aws_region=self.aws_region,
                   temperature=temperature,
                   top_K_OJ=top_k_oj,
                   top_K_govuk=top_k_govuk)

def get_args(parser):
    # Required argument for mode
    parser.add_argument('mode', type=str, choices=['interactive', 'test'],
                        help='The run mode: "interactive" for chat, or "test" for mass testing.')

    # Required argument for knowledge base path
    parser.add_argument('--kb_path', type=str, default=DEFAULT_KB_PATH,
                        help='Path to the knowledge base (e.g., CSV file) for RAG chunks.')

    # Optional argument for test data path (required only for 'test' mode)
    parser.add_argument('--test_data_path', type=str, default='./test_queries.json',
                        help='Path to the JSON/CSV file containing test queries and expected answers (required for "test" mode).')

    # Optional argument for overriding the AWS region
    parser.add_argument('--region', type=str, default="eu-west-2",
                        help=f'AWS region to use for the Bedrock client (default: eu-west-2).')

    parser.add_argument('--output_dir', type=str, help='Directory to save test outputs.')

    parser.add_argument('--role_arn', type=str, default=None, help='AWS Role ARN for Bedrock access (if required).')

    return parser.parse_args()
# Original command-line interface remains the entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"

    runner = AgentRunner(
        llm_model_id=model_id,
        path_to_kb=args.kb_path,
        path_to_test_data=args.test_data_path,
        aws_region=args.region,
        aws_role_arn=args.role_arn,
        output_dir=args.output_dir
    )

    if args.mode == 'test':

        oj_k, gov_k = 2, 5

        runner(args.mode, handoff_data=default_handoff(), top_k_oj=oj_k, top_k_govuk=gov_k)
    else:
        # Default interactive values
        runner(args.mode, handoff_data=default_handoff(), top_k_oj=3, top_k_govuk=3)
