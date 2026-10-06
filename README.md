# AI Powered Business — Customer Analytics & Churn Intelligence Platform

An end-to-end AI-powered business intelligence platform that combines customer analytics, machine learning, deep learning, NLP, computer vision, semantic search, Retrieval-Augmented Generation (RAG), and a local Large Language Model (LLM) to support data-driven business decisions.

## Project Overview

Businesses need to understand customer behavior, identify potential churn, analyze customer feedback, and make informed decisions.

This project demonstrates a complete analytics and AI workflow using a synthetic customer dataset containing 1,000 records.

The platform provides interactive business dashboards, customer churn predictions, natural-language customer search, and a GenAI assistant that retrieves relevant customer information before generating answers.

## Key Features

### 1. Business Intelligence Dashboard

* Total customer analysis
* Customer churn distribution
* Churn rate calculation
* Average customer satisfaction
* Support tickets versus churn analysis
* Satisfaction score versus churn analysis
* Interactive charts and customer data exploration

### 2. Machine Learning

Implemented and compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

Machine learning capabilities:

* Data preprocessing
* Feature scaling
* Train/test splitting
* Classification
* Model comparison
* Model evaluation
* Churn prediction
* Model persistence

### 3. Customer Churn Prediction

Predicts whether a customer is likely to churn based on customer attributes.

The application provides:

* Predicted churn status
* Estimated churn probability
* Low, medium, or high risk categorization
* Suggested customer retention actions

### 4. SQL Analytics

Demonstrates SQL-based business analysis, including:

* Total customer count
* Churn rate
* Churn by demographic groups
* Customer satisfaction analysis
* Support ticket analysis
* Website and app engagement
* High-risk customer segments
* Average customer profile

SQLite is used for SQL analysis.

### 5. Natural Language Processing (NLP)

Demonstrates text preprocessing and text feature extraction using:

* Text cleaning
* TF-IDF vectorization
* Logistic Regression-based text classification components

### 6. Deep Learning

Includes a CNN implemented with PyTorch.

Demonstrates:

* Convolutional layers
* ReLU activation
* Max pooling
* Dropout
* Fully connected layers
* Model training and evaluation
* Model saving and loading

**Important:** The CNN currently uses synthetic image data. Its accuracy should not be interpreted as evidence of real-world customer image classification performance.

### 7. Computer Vision

Includes image preprocessing and CNN-based image prediction.

Capabilities:

* Image loading
* Grayscale conversion
* Image resizing
* Pixel normalization
* CNN inference
* Prediction confidence output

The current computer vision component is a technical demonstration using synthetic training patterns, not a validated real-world vision model.

### 8. Embeddings and Semantic Search

Uses Sentence Transformers with the `all-MiniLM-L6-v2` model.

Demonstrates:

* Text embeddings
* 384-dimensional vector representations
* Semantic similarity
* Customer document retrieval
* Ranking relevant customer records

### 9. Retrieval-Augmented Generation (RAG)

The RAG component:

1. Converts customer records into text documents.
2. Generates text embeddings.
3. Retrieves relevant customer records using semantic similarity.
4. Supplies retrieved records as context for answering questions.

### 10. Generative AI Assistant

Uses the local `google/flan-t5-small` model through the Transformers library.

The assistant combines retrieved customer context with a language model to answer customer analytics questions.

Example questions:

* Which customers appear most at risk and why?
* Find customers with many support tickets.
* Identify customers with low satisfaction.
* Summarize relevant customer information.

The model can still produce incomplete or inaccurate answers, so generated answers should be checked against the retrieved evidence.

## Technology Stack

| Area             | Technologies                       |
| ---------------- | ---------------------------------- |
| Programming      | Python                             |
| Data Analysis    | Pandas, NumPy                      |
| Visualization    | Plotly, Streamlit                  |
| Machine Learning | Scikit-learn                       |
| Deep Learning    | PyTorch                            |
| Computer Vision  | Pillow, PyTorch                    |
| NLP              | TF-IDF, Scikit-learn               |
| Embeddings       | Sentence Transformers              |
| Generative AI    | Hugging Face Transformers, FLAN-T5 |
| Retrieval        | Embeddings, cosine similarity, RAG |
| Database         | SQLite, SQL                        |
| Model Storage    | Joblib, PyTorch                    |
| Interface        | Streamlit                          |

## Project Structure

```text
ai powered business/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── generate_data.py
│   └── customers.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── model_metadata.pkl
│   └── cnn_model.pth
│
└── src/
    ├── data_preprocessing.py
    ├── eda.py
    ├── feature_engineering.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    ├── sql_analysis.py
    │
    ├── nlp/
    │   ├── __init__.py
    │   ├── cnn_model.py
    │   └── train_cnn.py
    │
    ├── vision/
    │   ├── __init__.py
    │   ├── image_preprocessing.py
    │   └── image_prediction.py
    │
    └── genai/
        ├── __init__.py
        ├── embeddings.py
        ├── rag.py
        └── llm_assistant.py
```

## Installation

### Requirements

* Windows, Linux, or macOS
* Python 3.13 recommended for this project setup
* Internet access for downloading the embedding and language models the first time

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "ai powered business"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### Step 2: Install dependencies

```bash
py -3.13 -m pip install -r requirements.txt
```

On Linux or macOS, use the corresponding Python 3.13 executable if `py` is unavailable.

### Step 3: Generate the dataset if necessary

```bash
py -3.13 data/generate_data.py
```

### Step 4: Run the data processing workflow

```bash
py -3.13 src/data_preprocessing.py
py -3.13 src/eda.py
py -3.13 src/feature_engineering.py
```

### Step 5: Train and evaluate the churn model

```bash
py -3.13 src/train.py
py -3.13 src/evaluate.py
```

### Step 6: Run SQL analysis

```bash
py -3.13 src/sql_analysis.py
```

### Step 7: Run the CNN demonstration

```bash
py -3.13 src/nlp/train_cnn.py
```

### Step 8: Run embedding and RAG demonstrations

```bash
py -3.13 src/genai/embeddings.py
py -3.13 src/genai/rag.py
```

### Step 9: Launch the application

```bash
py -3.13 -m streamlit run app.py
```

Open the local URL displayed in the terminal.

**Note:** Train the models before launching the dashboard if the required model files do not exist.

## Model Evaluation

The current churn model evaluation produced the following results on the held-out test set:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 85.00% |
| Precision | 40.00% |
| Recall    |  6.90% |
| F1 Score  | 11.76% |
| ROC-AUC   | 71.28% |

These are results from the current synthetic dataset and model configuration.

### Interpretation

Although accuracy is 85%, the model detects only a small fraction of actual churn cases. The low recall means the current model misses many customers who churn.

Potential improvements include:

* Class weighting
* Decision threshold optimization
* Hyperparameter tuning
* Cross-validation
* Feature engineering
* Imbalance-aware evaluation

Accuracy alone is not sufficient to assess this churn model.

## Business Value

The platform demonstrates how analytics and AI can support:

* Early identification of potential churn
* Customer retention planning
* Customer segmentation
* Support workload analysis
* Customer satisfaction monitoring
* Faster customer information retrieval
* Natural-language business analysis

## Limitations

* The customer dataset is synthetic and does not represent verified real customer behavior.
* Churn model results are dataset-specific and require further improvement.
* The CNN uses synthetic images and is not validated for real-world computer vision tasks.
* The RAG component uses semantic similarity retrieval rather than a production vector database.
* The local language model can generate incomplete or incorrect responses.
* This project is a portfolio demonstration and is not validated for production business decisions.

## Future Improvements

* Deploy the application to a cloud platform.
* Add a production-grade vector database.
* Improve churn recall through threshold tuning and class imbalance techniques.
* Add automated testing and continuous integration.
* Introduce real, appropriately licensed datasets.
* Add monitoring, logging, and model versioning.
* Improve authentication and data security.

## Skills Demonstrated

Python, SQL, data analytics, statistics, machine learning, feature engineering, model evaluation, data visualization, Power BI-ready analytical thinking, PyTorch, CNNs, NLP, text embeddings, semantic search, RAG, LLM integration, Streamlit, and application development.

## Disclaimer

This project is intended for educational and portfolio purposes. The dataset is synthetic, and model outputs should not be treated as validated business predictions.

## Author

Mohammed Almas Qureshi

GitHub: https://github.com/mohd-almas786

LinkedIn: https://www.linkedin.com/in/mohammed-almas-qureshi-0abb61262
