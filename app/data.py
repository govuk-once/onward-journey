import pandas as pd
from sentence_transformers import SentenceTransformer

def df_to_text_chunks(df, chunk_size=5):
    """Converts a DataFrame into text chunks for embedding and retrieval.
    Args:
        df (pd.DataFrame): The DataFrame to convert.
        chunk_size (int): Number of rows per chunk.
    'Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    for _, row in df.iterrows():

        chunk = f"The unique id is {row['uid']}. The service name is {row['service_name']}. \
                  The department is {row['department']}. \
                  The phone number is {row['phone_number']}. \
                  The topic is {row['topic']}. \
                  The user type is {row['user_type']}. \
                  The tags are {row['tags']}. \
                  The url is {row['url']}.\
                  The last time the page was updated is {row['last_update']}.\
                  The description is {row['description']}."
        chunks.append(chunk)

    return chunks

class container:
    """
    Container class to load CSV data, process it into text chunks,
    and compute embeddings using a specified SentenceTransformer model.
    """
    def __init__(self, file_path: str,embedding_model: SentenceTransformer, chunk_size: int = 5):

        # Load CSV Data
        self.file_path       = file_path
        self.data            = pd.read_csv(self.file_path)

        # Process Data into Chunks and Embeddings
        self.chunk_data      = df_to_text_chunks(self.data, chunk_size=chunk_size)

        self.embedding_model = embedding_model
        self.embeddings      = self.embedding_model.encode(self.chunk_data)

    def get_embeddings(self):
        """Returns the computed global embeddings."""
        return self.embeddings

    def get_chunks(self):
        """Returns the original text chunks corresponding to the embeddings."""
        return self.chunk_data

    def get_embedding_model(self):
        """Returns the loaded SentenceTransformer model instance."""
        return self.embedding_model
