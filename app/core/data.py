import re 
import json
import time
import boto3
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic                 import BaseModel

def df_to_text_chunks(df):
    """Converts a DataFrame into text chunks for embedding and retrieval."""
    chunks = []
    for _, row in df.iterrows():
        chunk = (f"The unique id is {row['uid']}. The service name is {row['service_name']}. "
                 f"The department is {row['department']}. The phone number is {row['phone_number']}. "
                 f"The topic is {row['topic']}. The user type is {row['user_type']}. "
                 f"The tags are {row['tags']}. The url is {row['url']}. "
                 f"The last time the page was updated is {row['last_update']}. "
                 f"The description is {row['description']}.")
        chunks.append(chunk)
    return chunks

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

class vectorStore:
    """
    Updated Container class using Amazon Titan Text Embeddings V2.
    """
    def __init__(self, file_path: str, aws_region: str = 'eu-west-2'):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.chunk_data = df_to_text_chunks(self.data)

        # Initialize Bedrock client instead of loading a local model
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region
        )

        # Compute embeddings via API
        self.embeddings = self._generate_all_embeddings(self.chunk_data)

    def _get_single_embedding(self, text: str) -> List[float]:
        """Calls Bedrock API for a single chunk."""
        body = json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        })
        response = self.bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding')

    def _generate_all_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Processes chunks. Note: Titan V2 prefers individual or batch calls."""
        all_embeddings = []

        for i, chunk in enumerate(chunks):
            try:
                embedding = self._get_single_embedding(chunk)
                all_embeddings.append(embedding)
                # Small sleep to prevent ThrottlingException if CSV is massive
                if i % 10 == 0: time.sleep(0.1)
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                # Append zero-vector to maintain index alignment on failure
                all_embeddings.append([0.0] * 1024)

        return np.array(all_embeddings)

    def get_embeddings(self):
        return self.embeddings

    def get_chunks(self):
        return self.chunk_data
