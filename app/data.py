import pandas as pd
import boto3
import json
import numpy as np
import time
from typing import List

def df_to_text_chunks(df):
    """Converts a DataFrame into text chunks for embedding and retrieval."""
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
        chunk = (f"The unique id is {row['uid']}. The service name is {row['service_name']}. "
                 f"The department is {row['department']}. The phone number is {row['phone_number']}. "
                 f"The topic is {row['topic']}. The user type is {row['user_type']}. "
                 f"The tags are {row['tags']}. The url is {row['url']}. "
                 f"The last time the page was updated is {row['last_update']}. "
                 f"The description is {row['description']}.")
        chunks.append(chunk)
    return chunks

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
