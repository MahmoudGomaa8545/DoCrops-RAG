import logging
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import _faithfulness as faithfulness, _context_precision as context_precision, _context_recall as context_recall, _answer_relevancy as answer_relevancy
from langchain_openai import ChatOpenAI

# CONFIGURATION
OPENAI_API_KEY = "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"
HF_TOKEN = "your-hf-token-here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

from app import search_faiss, client

# Setup logging
logging.basicConfig(
    filename='rag_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Golden Dataset
test_questions = [
    {"question": "What causes chestnut blight?", "ground_truth": "Chestnut blight is caused by the fungus Cryphonectria parasitica."},
    {"question": "How is Dutch elm disease spread?", "ground_truth": "Dutch elm disease is carried to healthy elm trees by two elm bark beetles that deposit spores in the feeding wound."}
]

def get_system_response(query):
    logging.info(f"Retrieving and generating for query: {query}")
    matched_docs = search_faiss(query, k=3)
    context = "".join([f"{d['text']}\n\n" for d in matched_docs])
    
    prompt = f"Answer the user's question based on the context: {context}\n\nQuestion: {query}"
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    answer = response.choices[0].message.content.strip()
    return answer, [d['text'] for d in matched_docs]

def run_evaluation():
    logging.info("Starting live RAG evaluation...")
    data = []
    
    for q in test_questions:
        answer, contexts = get_system_response(q['question'])
        data.append({
            "question": q['question'],
            "ground_truth": q['ground_truth'],
            "answer": answer,
            "contexts": contexts
        })
    
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    evaluator_llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    try:
        results = evaluate(dataset, metrics=metrics, llm=evaluator_llm)
        logging.info(f"Evaluation results: {results}")
        print("Evaluation complete. Results:")
        print(results)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    run_evaluation()
