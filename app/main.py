from typing import Optional
import test
import random
import numpy as np
import torch
import argparse

from data                  import container
from agents                import OnwardJourneyAgent
from sentence_transformers import SentenceTransformer

AWS_MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"

def set_all_seeds(seed_value=42):
    """
    Sets the random seed for reproducibility across key libraries.
    """
    # 1. Python built-in 'random' module
    random.seed(seed_value)

    # 2. NumPy (used for array/matrix operations, including embeddings)
    np.random.seed(seed_value)

    # 3. PyTorch (the underlying framework for SentenceTransformer models)
    torch.manual_seed(seed_value)

    # 4. PyTorch GPU operations (if you are using a GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multiple GPUs

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return
def run_interactive_mode(path_to_knowledge_base, aws_role_arn: Optional[str], aws_region="eu-west-2", seed=1):
    """
    Main function to initialize the Onward Journey Agent and run a sample conversation
    (Interactive Mode - Based on the original main() logic).
    """
    # Set seeds for reproducibility
    set_all_seeds(seed)

    # Simulated handoff data for interactive mode
    simulated_handoff_data = {
        # Original sample handoff package
        "handoff_agent_id": "Chatbot Agent",
        # Updated model name
        "model_used": AWS_MODEL_ID,
        "system_instruction_used": "Generic assistant.",
        "final_conversation_history": [
            {"role": "user", "text": "I was asking about childcare."},
            {"role": "model", "text": "That requires specialized assistance."}],
        "next_agent_prompt": "Can you help me with childcare options?",
        "status": "DATA_COLLECTED_AND_READY_FOR_HANDOFF"
    }

    # Initialize Data Container
    cont = container(
        file_path=path_to_knowledge_base,
        embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
        chunk_size=5
    )
    # Initialize Onward Journey Agent
    oj_agent = OnwardJourneyAgent(
               handoff_package=simulated_handoff_data,
               embeddings=cont.get_embeddings(),
               chunk_data=cont.get_chunks(),
               embedding_model=cont.get_embedding_model(),
               model_name=AWS_MODEL_ID, # Use the Bedrock model ID
               aws_region=aws_region,   # Pass the AWS region
               aws_role_arn=aws_role_arn,
               temperature=0.0)

    # Run a sample conversation
    print("\n" + "-"*80)
    print("Starting Interactive Chat...")
    oj_agent.run_conversation()
    print("-"*80 + "\n")
    return
def run_test_mode(path_to_knowledge_base, test_data_path, aws_role_arn: Optional[str], aws_region="eu-west-2", seed=1):
    """
    Initializes the Onward Journey Agent for mass testing and runs the test suite.
    (Test Harness Mode).
    """

    # Set seeds for reproducibility
    set_all_seeds(seed)

    # Simulated handoff data for test harness mode
    simulated_handoff_data = {
        "handoff_agent_id": "Chatbot Agent",
        "model_used": AWS_MODEL_ID,
        "system_instruction_used": "Test mode.",
        "final_conversation_history": [],
        "next_agent_prompt": "Run Test Suite",
        "status": "TESTING_MODE"
    }

    # Initialize Data Container
    cont = container(
        file_path=path_to_knowledge_base,
        embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
        chunk_size=5
    )

    # Initialize Onward Journey Agent (using temperature 0.0 for deterministic test results)
    oj_agent = OnwardJourneyAgent(
               handoff_package=simulated_handoff_data,
               embeddings=cont.get_embeddings(),
               chunk_data=cont.get_chunks(),
               embedding_model=cont.get_embedding_model(),
               model_name=AWS_MODEL_ID,
               aws_region=aws_region,
               aws_role_arn=aws_role_arn,
               temperature=0.0)

    # Delegate the testing and analysis to the test suite in test.py
    test.run_mass_tests(oj_agent, test_data_path=test_data_path)

def main(run_mode, path_to_kb, test_data_path, aws_region_override=None, aws_role_arn: Optional[str] = None):

    region_to_use = aws_region_override if aws_region_override else "eu-west-2"

    print(f"AWS Region set to: {region_to_use}")

    if run_mode == 'interactive':
        print("Running in INTERACTIVE Mode.")
        run_interactive_mode(
            path_to_knowledge_base=path_to_kb,
            aws_region=region_to_use,
            aws_role_arn=aws_role_arn
        )
    elif run_mode == 'test':
        print("Running in MASS TESTING Mode.")
        run_test_mode(
            path_to_knowledge_base=path_to_kb,
            test_data_path=test_data_path,
            aws_region=region_to_use,
            aws_role_arn=aws_role_arn
        )
    else:
        print(f"Error: Unknown run_mode '{run_mode}'. Use 'test' or 'interactive'.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Onward Journey Agent in interactive or testing mode using AWS Bedrock.")

    # Required argument for mode
    parser.add_argument('mode', type=str, choices=['interactive', 'test'],
                        help='The run mode: "interactive" for chat, or "test" for mass testing.')

    # Required argument for knowledge base path
    parser.add_argument('--kb_path', type=str, required=True,
                        help='Path to the knowledge base (e.g., CSV file) for RAG chunks.')

    # Optional argument for test data path (required only for 'test' mode)
    parser.add_argument('--test_data', type=str, default='./test_queries.json',
                        help='Path to the JSON file containing test queries and expected answers (required for "test" mode).')

    # Optional argument for overriding the AWS region
    parser.add_argument('--region', type=str, default="eu-west-2",
                        help=f'AWS region to use for the Bedrock client (default: {"eu-west-2"}).')

    parser.add_argument('--role_arn', type=str, default=None, help="The ARN of the AWS role to assume for Bedrock calls if provided" )

    args = parser.parse_args()

    # Ensure test_data path is provided if the mode is 'test'
    if args.mode == 'test' and not args.test_data:
        parser.error("The --test_data argument is required when running in 'test' mode.")

    # Call the main func
    main(
        run_mode=args.mode,
        path_to_kb=args.kb_path,
        test_data_path=args.test_data,
        aws_region_override=args.region,
        aws_role_arn=args.role_arn
    )
