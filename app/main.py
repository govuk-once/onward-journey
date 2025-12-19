import argparse
import os
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import test
from agents import OnwardJourneyAgent
from data import vectorStore
from loaders import load_test_queries


# Calculate the default path to the mock data relative to this script (main.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KB_PATH = os.path.join(SCRIPT_DIR, "../mock_data/mock_rag_data.csv")


class AgentRunner:
    """
    Manages the configuration, setup, and execution modes (interactive and test)
    for the Onward Journey Agent, ensuring reproducibility and consistent setup.
    """

    def __init__(
        self,
        llm_model_id: str,
        path_to_kb: str,
        path_to_test_data: str,
        aws_region: str,
        aws_role_arn: str,
        output_dir: str,
        seed: int = 0,
        vector_store_model_id: str = "all-MiniLM-L6-v2",
        vector_store_chunk_size: int = 5,
    ):
        """
        Description: Initializes the manager with essential configuration parameters and sets seeds.

        llm_model_id          : The ID of the language model (e.g., Anthropic Claude).
        vector_store_model_id : The ID of the embedding model for the vector store.
        path_to_kb            : File path to the knowledge base document.
        path_to_test_data     : File path to the test data set.
        aws_region            : AWS region for model deployment.
        seed                  : Random seed for reproducibility.
        """

        # llm model id
        self.model_id = llm_model_id

        # Knowledge base parameters
        self.vector_store_model_id = vector_store_model_id
        self.vector_store_chunk_size = vector_store_chunk_size
        self.path_to_knowledge_base = path_to_kb

        # aws parameters
        self.aws_region = aws_region
        self.aws_role_arn = aws_role_arn

        # dir of test data
        self.path_to_test_data = path_to_test_data

        self.seed = seed
        self.output_dir = output_dir

        self._set_all_seeds(self.seed)

    def __call__(
        self, run_mode: str, handoff_data: dict, proto_test_name: str = "prototype_one"
    ):

        oj_agent = self._initialize_agent(handoff_data=handoff_data, temperature=0.0)

        if run_mode == "interactive":
            print("Running in INTERACTIVE Mode.")
            oj_agent.run_conversation()
        elif run_mode == "test":
            print("Running in TEST Mode.")
            print(f"Executing Test Suite from: {self.path_to_test_data}")
            test_queries = load_test_queries(self.path_to_test_data)
            if not test_queries:
                return
            getattr(test, f"{proto_test_name}_test")(
                oj_agent, test_queries=test_queries, output_dir=self.output_dir
            )
        else:
            print(f"Error: Unknown run_mode '{run_mode}'. Use 'test' or 'interactive'.")

        return

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
            torch.cuda.manual_seed_all(seed_value)  # For multiple GPUs

            # for deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _initialize_vector_store(self):
        """
        Returns the vector store of the onward journey knowledge base.
        """
        return vectorStore(
            file_path=self.path_to_knowledge_base,
            embedding_model=SentenceTransformer(self.vector_store_model_id),
            chunk_size=self.vector_store_chunk_size,
        )

    def _initialize_agent(
        self, handoff_data: dict, temperature: float
    ) -> OnwardJourneyAgent:
        """
        Initializes and returns the OnwardJourneyAgent with common configuration.
        """
        vector_store = self._initialize_vector_store()
        return OnwardJourneyAgent(
            handoff_package=handoff_data,
            vector_store_embeddings=vector_store.get_embeddings(),
            vector_store_embeddings_text_chunks=vector_store.get_chunks(),
            embedding_model=vector_store.get_embedding_model(),
            model_name=self.model_id,
            aws_region=self.aws_region,
            aws_role_arn=self.aws_role_arn,
            temperature=temperature,
        )


# Original command-line interface remains the entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Onward Journey Agent in interactive or testing mode using AWS Bedrock."
    )

    # Required argument for mode
    parser.add_argument(
        "mode",
        type=str,
        choices=["interactive", "test"],
        help='The run mode: "interactive" for chat, or "test" for mass testing.',
    )

    # Required argument for knowledge base path
    parser.add_argument(
        "--kb_path",
        type=str,
        default=DEFAULT_KB_PATH,
        help=f"Path to the knowledge base (default: {DEFAULT_KB_PATH}) for RAG chunks.",
    )

    # Optional argument for test data path (required only for 'test' mode)
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./test_queries.json",
        help='Path to the JSON/CSV file containing test queries and expected answers (required for "test" mode).',
    )

    # Optional argument for overriding the AWS region
    parser.add_argument(
        "--region",
        type=str,
        default="eu-west-2",
        help=f"AWS region to use for the Bedrock client (default: eu-west-2).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save test outputs.",
    )

    parser.add_argument(
        "--role_arn",
        type=str,
        default=None,
        help="AWS Role ARN for Bedrock access (if required).",
    )

    args = parser.parse_args()

    # Ensure test_data path is provided if the mode is 'test'
    if args.mode == "test" and not args.test_data_path:
        # This error case is mitigated by the default value, but kept for robustness
        # if the default were to be removed.
        parser.error(
            "The --test_data argument is required when running in 'test' mode."
        )

    # Model ID is hardcoded in the original script
    model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"

    # Initialize the Manager
    runner = AgentRunner(
        llm_model_id=model_id,
        path_to_kb=args.kb_path,
        path_to_test_data=args.test_data_path,
        aws_region=args.region,
        aws_role_arn=args.role_arn,
        output_dir=args.output_dir,
    )

    # Execute the objects call method with the specified mode
    runner(args.mode, handoff_data={}, proto_test_name="prototype_two")
