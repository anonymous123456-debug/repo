from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForSequenceClassification

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Model paths can be overridden without editing source code.
model2path = {
    "emb_model": os.environ.get("EMB_MODEL_PATH", os.path.join(PROJECT_ROOT, "multilingual-e5-large")),
    "rerank_model": os.environ.get("RERANK_MODEL_PATH", os.path.join(PROJECT_ROOT, "rerank")),
}
device = "cuda" if torch.cuda.is_available() else "cpu"
emb_model = SentenceTransformer(model2path["emb_model"]).to(device)
cross_tokenizer = AutoTokenizer.from_pretrained(model2path["rerank_model"])
cross_model = AutoModelForSequenceClassification.from_pretrained(model2path["rerank_model"]).to(device)

# Retrieval settings.
class Args:
    top_k1 =20
    top_k2 =7
args = Args()

# Flask App
app = Flask(__name__)

# Core retrieval functions.
def vector_search(question, chunk_data,vector_path):
    vector = faiss.read_index(vector_path)
    feature = emb_model.encode([question])
    distance, match_id = vector.search(np.array(feature), args.top_k1)
    content = [chunk_data[int(i)] for i in match_id[0]]
    return content, list(match_id[0])

def sort_section(question, section, match_id,dataset):
    q = [question] * len(section)
    if dataset == 'medqa':
        section = [item["text"] for item in section if "text" in item]
    features = cross_tokenizer(q, section, padding=True, truncation=True, return_tensors="pt").to(device)
    cross_model.eval()
    with torch.no_grad():
        scores = cross_model(**features).logits.squeeze(dim=1)
    sort_scores = torch.argsort(scores, dim=0, descending=True).cpu()
    result = [section[sort_scores[i].item()] for i in range(args.top_k2)]
    match_id = [match_id[sort_scores[i].item()] for i in range(args.top_k2)]
    return result, match_id

# API route for first-stage vector retrieval.
@app.route('/vector_search', methods=['POST'])
def handle_vector_search():
    data = request.get_json()
    question = data.get("question")
    chunk_data = data.get("chunk_data")
    vector_path=data.get("vector_path")
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        content, ids = vector_search(question, chunk_data,vector_path)
        return jsonify({
            "chunks": content,
            "ids": [int(i) for i in ids] 
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API route for reranking retrieved sections.
@app.route('/sort_section', methods=['POST'])
def handle_sort_section():
    data = request.get_json()
    question = data.get("question")
    section = data.get("section")
    match_id = data.get("match_id")
    dataset  =data.get("dataset")
    if not question or not section or not match_id:
        return jsonify({"error": "Missing 'question', 'section', or 'match_id' field"}), 400
    try:
        result, sorted_match_id = sort_section(question, section, match_id,dataset)
        return jsonify({
            "result": result,
            "match_id": sorted_match_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9931)
