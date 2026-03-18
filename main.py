import os
import json
import random
import asyncio
import argparse
import numpy as np
from datetime import datetime, timezone

from app.core.engine import CAGQueryCache
from app.evaluation.test import Evaluator
# Internal package imports
from app.integrations.genesys import GenesysServiceDiscovery
from app.agents import OnwardJourneyAgent, GovUKAgent, hybridAgent
from app.core.data import LocalCSVVectorStore, GenesysCloudVectorStore
from app.evaluation.benchmarking import load_test_queries, clarification_success_gain_metric

from dotenv import load_dotenv

load_dotenv()

# Constants for default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KB_PATH = os.path.join(SCRIPT_DIR, "app", "resources", "data", "oj_rag_data.csv")
DEFAULT_CAG_PATH = os.path.join(SCRIPT_DIR, "app", "resources", "data", "cag_interaction.json")

class AgentRunner:
    def __init__(self, args, model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"):
        self.args = args
        self.model_id = model_id
        self._set_seeds(args.seed)

        # Initialize Cache if enabled
        self.cache = CAGQueryCache(args.cag_cache_file_path) if args.cag_cache else None

    def _set_seeds(self, seed: int):
        """Standardize randomness for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def _get_agent(self, vs_oj: LocalCSVVectorStore, vs_gen: GenesysCloudVectorStore, handoff_data: dict):
        """Factory to initialize the requested agent type with correct Top-K values."""
        agent_mapping = {
            0: OnwardJourneyAgent,
            1: GovUKAgent,
            2: hybridAgent
        }

        agent_config = {
            "model_name" : self.model_id,
            "aws_region" : self.args.region,
            "temperature": 0.0,
            "handoff_package": handoff_data,
            # data stores
            "vector_store_embeddings" : vs_oj.get_embeddings(),
            "vector_store_chunks": vs_oj.get_chunks(),
            "genesys_embeddings" : vs_gen.get_embeddings(),
            "genesys_chunks": vs_gen.get_chunks(),
            # hyperparameters
            "top_K_OJ": self.args.top_k_oj,
            "top_K_govuk": self.args.top_k_govuk
        }

        agent_cls = agent_mapping.get(self.args.agent_type, OnwardJourneyAgent)

        return agent_cls(**agent_config)
    
    async def run_interactive(self):
        """Terminal loop for real-time interaction."""
        vs_oj = LocalCSVVectorStore(file_path=self.args.kb_path)
        
        discover = GenesysServiceDiscovery()
        raw_gen_data = discover.get_all_kb_content(os.getenv("GENESYS_KB_ID"))
        vs_gen = GenesysCloudVectorStore(raw_gen_data)

        agent = self._get_agent(vs_oj, vs_gen, {'final_conversation_history': []})

        print("\n" + "="*60)
        print(" AGENT (Interactive Mode) ")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit", "end"]:
                    print("\n Conversation with Agent ended.👋")
                    break
                if not user_input:
                    continue

                #Cache lookup (CAG)
                if self.cache:
                    match = self.cache.lookup(user_input, threshold=self.args.cag_cache_threshold)
                    if match:
                        print(f"\nAgent (Cache Hit - Score {match.score:.2f}):\n{match.answer}\n")
                        continue

                # LLM / Tool execution
                response = await agent._send_message_and_tools(user_input)
                print(f"\nAgent:\n{response}\n")

                # CAG collection
                if self.args.cag_collect and self._is_helpful():
                    self._save_interaction(user_input, response)

            except KeyboardInterrupt:
                break

    def _is_helpful(self) -> bool:
        """User feedback loop for cache collection."""
        return input("Was this answer helpful? (y/n): ").lower().startswith('y')

    def _save_interaction(self, query: str, answer: str):
        """Appends accepted interactions to the JSON cache."""
        path = self.args.cag_file_path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer
        }

        records = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    records = json.load(f)
                except:
                    records = []

        records.append(record)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Interaction cached to {path} 💾")

    def run_test(self):
        """Batch evaluation mode."""
        vs = LocalCSVVectorStore(file_path=self.args.kb_path)
        agent = self._get_agent(vs, {'final_conversation_history': []})

        pair_folder = f"oj{self.args.top_k_oj}_gov{self.args.top_k_govuk}"
        run_dir = os.path.join(self.args.output_dir, pair_folder)
        os.makedirs(run_dir, exist_ok=True)

        test_queries = load_test_queries(self.args.test_data_path)
        evaluator = Evaluator(agent, test_queries, run_dir)

        print(f"Running Forced Mode...")
        forced_df = evaluator('forced')

        print(f"Running Clarification Mode...")
        clarification_df = evaluator('clarification')

        metrics = clarification_success_gain_metric(clarification_df, forced_df)
        print(f"\nTest Complete. CSG Score: {metrics.get('clarification_success_gain_csg', 0):.4f}")

def get_args():
    parser = argparse.ArgumentParser(description="GOV.UK Agent Runner")
    parser.add_argument('mode', choices=['interactive', 'test'], help='Run mode')

    # Paths
    parser.add_argument('--kb_path', default=DEFAULT_KB_PATH)
    parser.add_argument('--test_data_path', default='./test_queries.json')
    parser.add_argument('--output_dir', default='./test_output')

    # AWS/Agent Config
    parser.add_argument('--region', default="eu-west-2")
    parser.add_argument('--agent_type', type=int, default=0, help='0: OJ, 1: GovUK, 2: Hybrid')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_k_oj', type=int, default=3)
    parser.add_argument('--top_k_govuk', type=int, default=3)

    # CAG (Cache) Config
    parser.add_argument('--cag_collect', action='store_true', help='Enable user feedback collection')
    parser.add_argument('--cag_file_path', default='cag_interaction.json')
    parser.add_argument('--cag_cache', action='store_true', help='Enable cache lookup')
    parser.add_argument('--cag_cache_threshold', type=float, default=0.92)
    parser.add_argument('--cag_cache_file_path', default=DEFAULT_CAG_PATH)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    runner = AgentRunner(args)

    if args.mode == 'interactive':
        asyncio.run(runner.run_interactive())
    else:
        runner.run_test()
