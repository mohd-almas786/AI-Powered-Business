import os
import sys
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Project paths

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from rag import CustomerRAG


# Local LLM Configuration

MODEL_NAME = "google/flan-t5-small"


class GenAICustomerAssistant:

    def __init__(self):

        print("Loading RAG system...")

        self.rag = CustomerRAG()

        print("\nLoading local LLM...")
        print(f"Model: {MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        self.model.eval()

        print("LLM loaded successfully.")

    # Retrieve relevant customer information

    def retrieve_context(self, question, top_k=5):

        results = self.rag.search(
            question,
            top_k=top_k
        )

        context = ""

        for i, result in enumerate(results, start=1):

            context += (
                f"\nCustomer {i}:\n"
                f"{result['document']}\n"
                f"Similarity: {result['similarity']:.4f}\n"
            )

        return results, context

    # Generate LLM response

    def generate_answer(self, question):

        results, context = self.retrieve_context(
            question,
            top_k=5
        )

        prompt = f"""
You are an AI-powered business customer analytics assistant.

Answer the user's question using ONLY the customer information
provided in the context.

Do not invent customer information.

If the context is insufficient, say:
"Insufficient information in the retrieved customer data."

Context:
{context}

Question:
{question}

Provide a short, clear business-oriented answer.
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.3,
                do_sample=True
            )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer, results


# Main program

if __name__ == "__main__":

    print("=" * 60)
    print("AI POWERED CUSTOMER ANALYTICS ASSISTANT")
    print("=" * 60)

    assistant = GenAICustomerAssistant()

    print("\nExample questions:")
    print("1. Which customers appear most at risk?")
    print("2. Which customers have many support complaints?")
    print("3. Find unhappy customers who may churn.")
    print("4. Which customers have low satisfaction?")
    print("5. Which customers have high support tickets?")

    question = input(
        "\nEnter your business question: "
    ).strip()

    if not question:

        print("Please enter a question.")

    else:

        print("\nSearching customer knowledge base...")

        answer, results = assistant.generate_answer(
            question
        )

        print("\n" + "=" * 60)
        print("GENAI RESPONSE")
        print("=" * 60)

        print(answer)

        print("\n" + "=" * 60)
        print("RETRIEVED CUSTOMER RECORDS")
        print("=" * 60)

        for i, result in enumerate(results, start=1):

            print(
                f"\n{i}. Customer ID: "
                f"{result['customer_id']}"
            )

            print(
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )

            print(
                result["document"]
            )

    print("\n" + "=" * 60)
    print("GenAI assistant completed successfully!")
    print("=" * 60)