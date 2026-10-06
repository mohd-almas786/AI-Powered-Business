import os
import sys
import numpy as np
import pandas as pd

# Allow importing embeddings.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from embeddings import CustomerEmbeddingModel


class CustomerRAG:
    """
    Simple Retrieval-Augmented Generation retrieval system.

    This component:
    1. Loads customer data
    2. Converts customer records into text
    3. Creates embeddings
    4. Retrieves the most relevant customers
    """

    def __init__(self, data_path="data/customers.csv"):

        print("Initializing Customer RAG system...")

        self.data_path = data_path

        self.embedding_model = CustomerEmbeddingModel()

        self.documents = []
        self.embeddings = None

        self.load_data()

    # Load Customer Data

    def load_data(self):

        if not os.path.exists(self.data_path):

            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        self.data = pd.read_csv(
            self.data_path
        )

        print(
            "Customer dataset loaded:",
            self.data.shape
        )

        self.create_documents()

        self.create_embeddings()

    # Convert Customer Records into Text

    def create_documents(self):

        self.documents = []

        for _, row in self.data.iterrows():

            document = (
                f"Customer ID: {row['customer_id']}. "
                f"Age: {row['age']}. "
                f"Gender: {row['gender']}. "
                f"Income: {row['income']}. "
                f"Tenure: {row['tenure_months']} months. "
                f"Support tickets: {row['support_tickets']}. "
                f"Website visits: {row['website_visits']}. "
                f"App usage: {row['app_usage_hours']} hours. "
                f"Satisfaction score: "
                f"{row['satisfaction_score']}. "
                f"Churn status: {row['churn']}."
            )

            self.documents.append(
                document
            )

        print(
            "Documents created:",
            len(self.documents)
        )

    # Create Embeddings

    def create_embeddings(self):

        print(
            "\nCreating customer embeddings..."
        )

        self.embeddings = (
            self.embedding_model.generate_embeddings(
                self.documents
            )
        )

        print(
            "Embedding matrix shape:",
            self.embeddings.shape
        )

    # Similarity Search

    def search(self, query, top_k=5):

        query_embedding = (
            self.embedding_model.generate_embedding(
                query
            )
        )

        # Normalize vectors
        document_norms = np.linalg.norm(
            self.embeddings,
            axis=1
        )

        query_norm = np.linalg.norm(
            query_embedding
        )

        similarities = np.dot(
            self.embeddings,
            query_embedding
        ) / (
            document_norms * query_norm + 1e-10
        )

        # Get highest similarity scores
        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append(
                {
                    "customer_id": int(
                        self.data.iloc[index][
                            "customer_id"
                        ]
                    ),
                    "similarity": float(
                        similarities[index]
                    ),
                    "document": self.documents[
                        index
                    ]
                }
            )

        return results


# Demo

if __name__ == "__main__":

    rag = CustomerRAG()

    query = (
        "customers who are unhappy "
        "and have many support complaints"
    )

    print("\n--------------------------------")
    print("RAG Semantic Search")
    print("--------------------------------")

    print("Query:")
    print(query)

    results = rag.search(
        query,
        top_k=5
    )

    print("\nTop relevant customers:\n")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. Customer "
            f"{result['customer_id']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            result["document"]
        )

        print("-" * 70)

    print(
        "\nRAG retrieval completed successfully!"
    )