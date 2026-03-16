import re 
import json
import time
import boto3
import numpy as np
import pandas as pd
from abc import ABC
from typing import List, Optional, Dict, Any
from pydantic                 import BaseModel


def extract_and_standardize_phone(text: str) -> str:
    """
    Tries to extract a UK phone number and standardizes the format
    (e.g., '0300 200 3887') to match the expected class labels.
    """

    # Pattern 1: Common non-geographic/mobile-like split (e.g., 4-3-X)
    pattern_4_3_X = r'\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4}'

    # Pattern 2: Common freephone/geographic split (e.g., 4-2-2-2)
    pattern_4_2_2_2 = r'\d{4}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}'

    # Combine the patterns
    combined_pattern = r'\b(' + pattern_4_3_X + r'|' + pattern_4_2_2_2 + r')\b'

    match = re.search(combined_pattern, text)
    if match:
        # 1. Clean up: remove spaces and hyphens
        extracted_num_cleaned = match.group(1).replace(' ', '').replace('-', '')

        # 2. Re-format to the standard output format (4-3-X for consistency)
        if len(extracted_num_cleaned) >= 10:
             return extracted_num_cleaned[0:4] + ' ' + extracted_num_cleaned[4:7] + ' ' + extracted_num_cleaned[7:]

        return ' '.join(extracted_num_cleaned[i:i+3] for i in range(0, len(extracted_num_cleaned), 3)).strip()

    return 'NOT_FOUND' # Consistent misclassification label

class SearchResult(BaseModel):
    url: str
    score: float
    document_type: str
    title: str
    description: Optional[str]
    heading_hierarchy: list[str]
    html_content: str

class BaseVectorStore(ABC):

    def __init__(self, aws_region: str = 'eu-west-2', dimensions: int = 1024):

        self.bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=aws_region)
        
        self.dimensions = dimensions 
        self.chunk_data: List[str] = []
        self.embeddings: np.ndarray = np.array([])

    def _get_single_embedding(self, text: str) -> List[float]:
        """Bedrock API call for Titan V2"""

        body = json.dumps({"inputText": text, "dimensions": self.dimensions, "normalize": True})

        try:
            response = self.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            return json.loads(response.get('body').read()).get('embedding', [])
        except Exception as e:
            print(f"Embedding API Error: {e}")
            return [0.0] * self.dimensions 
    def _generate_embeddings(self):
        """Standardized loop"""

        all_vecs = []
        for i, chunk in enumerate(self.chunk_data):
            all_vecs.append(self._get_single_embedding(chunk))
            if i % 10 == 0:
                time.sleep(1)
        
        self.embeddings = np.array(all_vecs)
    
    def get_embeddings(self) -> np.ndarray:
        return self.embeddings
    
    def get_chunks(self) -> List[str]:
        return self.chunk_data 
    
class LocalCSVVectorStore(BaseVectorStore):
    """Handles OJ Knowledge Base"""
    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
        df = pd.read_csv(file_path)
        self.chunk_data = self._process_df(df)
        self._generate_embeddings()

    def _process_df(self, df: pd.DataFrame) -> List[str]:
        return [
                    (f"The unique ID is: {r['uid']}. The service name is: {r['service_name']}. The department is: {r['department']}. "
                    f"The phone number is: {r['phone_number']}. The topic is topic: {r['topic']}. The tags are: {r['tags']}. "
                    f"The URL is: {r['url']}. The last time the page was updated is {r['last_update']}. The description is: {r['description']}")
                    for _, r in df.iterrows()
                ]

class GenesysCloudVectorStore(BaseVectorStore):
    """Handles flattened Genesys Knowledge Base API data"""
    def __init__(self, raw_data: List[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.chunk_data = [f"Source: Genesys Cloud Knowledge Base. \
                           Title: {item['title']}. \
                           Content: {item['content']}"
                           for item in raw_data]
        self._generate_embeddings()

    
