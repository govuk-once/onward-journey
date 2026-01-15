import random
import numpy as np
import torch
import argparse 

from data                        import vectorStore
from agents                      import OnwardJourneyAgent 
from loaders                     import load_test_queries
from metrics                     import clarification_success_gain_metric
import test 

def default_handoff():
    return {'handoff_agent_id': 'GOV.UK Chat', 'final_conversation_history': []}

class AgentRunner:
    """
    Manages the configuration, setup, and execution modes (interactive and test)
    for the Onward Journey Agent, ensuring reproducibility and consistent setup.
    """

    def __init__(self, llm_model_id: str, path_to_kb: str, path_to_test_data: str, 
                 aws_region: str, aws_role_arn : str, output_dir: str,  
                 seed: int = 0, vector_store_model_id: str = 'amazon.titan-embed-text-v2:0'):
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
        self.model_id                  = llm_model_id

        # Knowledge base parameters
        self.vector_store_model_id     = vector_store_model_id
        self.path_to_knowledge_base    = path_to_kb

        # aws parameters
        self.aws_region                = aws_region
        self.aws_role_arn              = aws_role_arn

        # dir of test data
        self.path_to_test_data         = path_to_test_data

        self.seed                      = seed
        self.output_dir                = output_dir

        self._set_all_seeds(self.seed)
    
    def __call__(self, run_mode: str, handoff_data: dict, proto_test_name: str = "prototype_one"):

        vs = self._initialize_vector_store()

        oj_agent = self._initialize_agent(vs=vs, handoff_data=handoff_data, temperature=0.0)

        if run_mode == 'interactive':
            print("Running in INTERACTIVE Mode.")
            oj_agent.run_conversation()
        elif run_mode == 'test':
            print('Running in TEST Mode.')
            print(f"Executing Test Suite from: {self.path_to_test_data}")
            test_queries = load_test_queries(self.path_to_test_data)

            # Initialize the new Evaluator class
            evaluator = test.Evaluator(oj_agent, test_queries, self.output_dir)

            # 1. Execute standardized trials via the Evaluator
            print("Executing Forced Mode Evaluation...")
            forced_df        = evaluator('forced')
            
            print("Executing Clarification Mode Evaluation...")
            clarification_df = evaluator('clarification')

            # 2. Calculate Clarification Success Gain (CSG)
            gain_metrics =  clarification_success_gain_metric(clarification_df, forced_df)
            
            print("\n" + "="*50)
            print("QUANTITATIVE PERFORMANCE SUMMARY")
            print(f"Clarification Accuracy: {gain_metrics.get('ts_clarity_accuracy', 0):.2%}")
            print(f"Forced Accuracy:        {gain_metrics.get('ts_initial_accuracy', 0):.2%}")
            print(f"CSG Score:              {gain_metrics.get('clarification_success_gain_csg', 0):.4f}")
            print("="*50 + "\n")



            if not test_queries: 
                return
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
            torch.cuda.manual_seed_all(seed_value) # For multiple GPUs
            
            # for deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def _initialize_vector_store(self):
        """
        Returns the vector store of the onward journey knowledge base.
        """
        return vectorStore(
            file_path=self.path_to_knowledge_base,
        )
    def _initialize_agent(self, vs: vectorStore, handoff_data: dict, temperature: float) -> OnwardJourneyAgent:
        """
        Initializes and returns the OnwardJourneyAgent with common configuration.
        """
        return OnwardJourneyAgent(
                   handoff_package=handoff_data,
                   vector_store_embeddings=vs.get_embeddings(),
                   vector_store_chunks=vs.get_chunks(),
                   embedding_model=self.vector_store_model_id ,
                   model_name=self.model_id,
                   aws_region=self.aws_region,
                   temperature=temperature)


def get_args(parser):
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

    return parser.parse_args()
# Original command-line interface remains the entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Onward Journey Agent in interactive or testing mode using AWS Bedrock.")
    args = get_args(parser)

    # Ensure test_data path is provided if the mode is 'test'
    if args.mode == 'test' and not args.test_data_path:
        # This error case is mitigated by the default value, but kept for robustness 
        # if the default were to be removed.
        parser.error("The --test_data argument is required when running in 'test' mode.")

    model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    # Initialize the Agent manager and run
    runner = AgentRunner(
        llm_model_id=model_id,
        path_to_kb=args.kb_path,
        path_to_test_data=args.test_data_path,
        aws_region=args.region,
        aws_role_arn = args.role_arn,
        output_dir=args.output_dir
    )

    runner(args.mode, handoff_data=default_handoff(), proto_test_name="prototype_two")