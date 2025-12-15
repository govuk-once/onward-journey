import random
import numpy as np
import torch
import argparse

from data                  import container
from agents                import OnwardJourneyAgent
from loaders               import load_test_queries
from sentence_transformers import SentenceTransformer

import test

class AgentRunner:
    """
    Manages the configuration, setup, and execution modes (interactive and test)
    for the Onward Journey Agent, ensuring reproducibility and consistent setup.
    """

    DEFAULT_REGION       = "eu-west-2"
    DEFAULT_SEED         = 1
    EMBEDDING_MODEL_ID   = 'all-MiniLM-L6-v2'
    KNOWLEDGE_CHUNK_SIZE = 5

    def __init__(self, model_id: str, path_to_kb: str, path_to_test_data: str,
                 aws_region: str, aws_role_arn : str, output_dir: str,
                 seed: int = DEFAULT_SEED):
        """
        Description: Initializes the manager with essential configuration parameters and sets seeds.

        model_id          : The ID of the language model (e.g., Anthropic Claude).
        path_to_kb        : File path to the knowledge base document.
        path_to_test_data : File path to the test data set.
        aws_region        : AWS region for model deployment.
        seed              : Random seed for reproducibility.
        """
        self.model_id               = model_id
        self.path_to_knowledge_base = path_to_kb
        self.path_to_test_data      = path_to_test_data
        self.aws_region             = aws_region
        self.aws_role_arn           = aws_role_arn
        self.seed                   = seed

        self.output_dir             = output_dir

        self._set_all_seeds(self.seed)

    def _set_all_seeds(self, seed_value: int):
        """
        Sets the random seed for reproducibility across key libraries (Python, NumPy, PyTorch).
        """
        # Python built-in 'random' module
        random.seed(seed_value)

        # NumPy (used for array/matrix operations)
        np.random.seed(seed_value)

        # PyTorch
        torch.manual_seed(seed_value)

        # PyTorch GPU operations (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) # For multiple GPUs

            # for deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _initialize_data_container(self):
        """
        Initializes and returns the Data Container for knowledge base access.
        """
        cont = container(
            file_path=self.path_to_knowledge_base,
            embedding_model=SentenceTransformer(self.EMBEDDING_MODEL_ID),
            chunk_size=self.KNOWLEDGE_CHUNK_SIZE
        )
        return cont

    def _initialize_agent(self, handoff_data: dict, temperature: float) -> OnwardJourneyAgent:
        """
        Initializes and returns the OnwardJourneyAgent with common configuration.
        """
        cont = self._initialize_data_container()

        # Initialize Onward Journey Agent
        oj_agent = OnwardJourneyAgent(
                   handoff_package=handoff_data,
                   embeddings=cont.get_embeddings(),
                   chunk_data=cont.get_chunks(),
                   embedding_model=cont.get_embedding_model(),
                   model_name=self.model_id,
                   aws_region=self.aws_region,
                   aws_role_arn=self.aws_role_arn,
                   temperature=temperature)

        return oj_agent

    def run_interactive_mode(self, handoff_data: dict = {}):
        """
        Initializes the Agent for interactive chat and runs a sample conversation.
        Corresponds to the original run_interactive_mode function.
        """
        print("Running in INTERACTIVE Mode.")

        # Use temperature 0.0 for deterministic RAG retrieval, consistent with the original code
        oj_agent = self._initialize_agent(handoff_data, temperature=0.0)

        # Run a sample conversation
        print("\n" + "-"*80)
        print("Starting Interactive Chat...")
        oj_agent.run_conversation()
        print("-"*80 + "\n")

    def run_test_mode(self, handoff_data: dict = {}, proto_test_name: str = "prototype_one"):
        """
        Initializes the Agent for mass testing and executes the test harness.
        Corresponds to the original run_test_mode function.
        """
        print("Running in MASS TESTING Mode.")

        # Use temperature 0.0 for deterministic test results
        oj_agent = self._initialize_agent(handoff_data=handoff_data, temperature=0.0)

        # Delegate the testing and analysis
        print("\n" + "-"*80)
        print(f"Executing Test Suite from: {self.path_to_test_data}")
        print("-"*80)

        test_queries = load_test_queries(self.path_to_test_data)
        if not test_queries:
            return

        getattr(test, f"{proto_test_name}_test")(oj_agent, test_queries=test_queries, output_dir=self.output_dir)

    def execute(self, run_mode: str, handoff_data: dict,  proto_test_name: str = "prototype_one"):
        """
        The main execution method, dispatching to the appropriate run mode.
        Corresponds to the original main function logic.
        """
        if run_mode == 'interactive':
            self.run_interactive_mode(handoff_data)
        elif run_mode == 'test':
            self.run_test_mode(handoff_data, proto_test_name=proto_test_name)
        else:
            # Should not be reached if argparse is configured correctly, but good for safety
            print(f"Error: Unknown run_mode '{run_mode}'. Use 'test' or 'interactive'.")

        return

# Original command-line interface remains the entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Onward Journey Agent in interactive or testing mode using AWS Bedrock.")

    # Required argument for mode
    parser.add_argument('mode', type=str, choices=['interactive', 'test'],
                        help='The run mode: "interactive" for chat, or "test" for mass testing.')

    # Required argument for knowledge base path
    parser.add_argument('--kb_path', type=str, required=True,
                        help='Path to the knowledge base (e.g., CSV file) for RAG chunks.')

    # Optional argument for test data path (required only for 'test' mode)
    parser.add_argument('--test_data_path', type=str, default='./test_queries.json',
                        help='Path to the JSON/CSV file containing test queries and expected answers (required for "test" mode).')

    # Optional argument for overriding the AWS region
    parser.add_argument('--region', type=str, default="eu-west-2",
                        help=f'AWS region to use for the Bedrock client (default: eu-west-2).')

    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save test outputs.')

    parser.add_argument('--role_arn', type=str, default=None, help='AWS Role ARN for Bedrock access (if required).')

    args = parser.parse_args()

    # Ensure test_data path is provided if the mode is 'test'
    if args.mode == 'test' and not args.test_data_path:
        # This error case is mitigated by the default value, but kept for robustness
        # if the default were to be removed.
        parser.error("The --test_data argument is required when running in 'test' mode.")

    # Model ID is hardcoded in the original script
    model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"

    # Initialize the Manager
    runner = AgentRunner(
        model_id=model_id,
        path_to_kb=args.kb_path,
        path_to_test_data=args.test_data_path,
        aws_region=args.region,
        aws_role_arn = args.role_arn,
        output_dir=args.output_dir
    )

    # Execute the Manager's main logic
    runner.execute(args.mode, handoff_data={}, proto_test_name="prototype_two")
