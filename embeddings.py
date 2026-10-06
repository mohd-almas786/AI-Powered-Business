from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"


class CustomerEmbeddingModel:
    """
    Converts customer-related text into numerical embeddings.
    """

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("Embedding model loaded successfully.")

    def generate_embedding(self, text):
        """
        Convert one text into an embedding vector.
        """

        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding

    def generate_embeddings(self, texts):
        """
        Convert multiple texts into embedding vectors.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings


if __name__ == "__main__":

    embedding_model = CustomerEmbeddingModel()

    sample_texts = [
        "Customer is highly satisfied with the service.",
        "Customer has many support complaints.",
        "Customer is considering leaving the company."
    ]

    embeddings = embedding_model.generate_embeddings(
        sample_texts
    )

    print("\nEmbedding demonstration")
    print("-----------------------")

    print("Number of texts:", len(sample_texts))

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Single embedding dimensions:",
        embeddings.shape[1]
    )

    print(
        "\nFirst 10 values of first embedding:"
    )

    print(
        np.round(
            embeddings[0][:10],
            4
        )
    )