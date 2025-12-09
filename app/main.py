import test
import random
import numpy as np
import torch

from data                  import container
from agents                import OnwardJourneyAgent
from sentence_transformers import SentenceTransformer

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

# --- INTERACTIVE MODE ---
def run_interactive_mode(path_to_knowledge_base, api_key, seed=1):
    """
    Main function to initialize the Onward Journey Agent and run a sample conversation
    (Interactive Mode - Based on the original main() logic).
    """

    # Set seeds for reproducibility
    set_all_seeds(seed)

    simulated_handoff_data = {
        # Original sample handoff package
        "handoff_agent_id": "Chatbot Agent",
        "model_used": "gemini-2.5-flash",
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
               model_name='gemini-2.5-flash',
               api_key=api_key,
               temperature=0.0)

    # Run a sample conversation (Assuming run_interactive_chat method exists in agents.py)
    print("\n" + "-"*80)
    print("Starting Interactive Chat...")
    oj_agent.run_conversation()
    print("-"*80 + "\n")

# --- TEST MODE ---
def run_test_mode(path_to_knowledge_base, test_data_path, api_key, seed=1):
    """
    Initializes the Onward Journey Agent for mass testing and runs the test suite.
    (Test Harness Mode).
    """

    # Set seeds for reproducibility
    set_all_seeds(seed)

    simulated_handoff_data = {
        # Handoff package structured for testing
        "handoff_agent_id": "Test Harness",
        "model_used": "gemini-2.5-flash",
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
               model_name='gemini-2.5-flash',
               api_key=api_key,
               temperature=0.0)

    # Delegate the testing and analysis to the test.py file
    test.run_mass_tests(oj_agent, test_data_path=test_data_path)

def main(RUN_MODE, PATH_TO_KB, TEST_DATA_PATH, API_KEY):
    if RUN_MODE == 'interactive':
        print("Running in INTERACTIVE Mode.")
        run_interactive_mode(path_to_knowledge_base=PATH_TO_KB, api_key=API_KEY)
    elif RUN_MODE == 'test':
        print("Running in MASS TESTING Mode.")
        run_test_mode(path_to_knowledge_base=PATH_TO_KB, test_data_path=TEST_DATA_PATH, api_key=API_KEY)
    else:
        print(f"Error: Unknown RUN_MODE '{RUN_MODE}'. Use 'test' or 'interactive'.")
    return

if __name__ == "__main__":
    main(
        RUN_MODE='interactive',  # Change to 'interactive' for interactive mode
        PATH_TO_KB='knowledge_base.csv',
        TEST_DATA_PATH='test_queries.csv',
        API_KEY='AIzaSyCSQmSu9n89CQ7UgDrfZYPcPisKuthslx8')
