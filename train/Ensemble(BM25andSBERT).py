import json
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import nltk

# Tải bộ dữ liệu cần thiết cho NLTK
nltk.download('punkt_tab')

# Load mô hình SBERT
sbert_model = SentenceTransformer('all-mpnet-base-v2')

# Đọc nội dung từ file JSON
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Chuyển dữ liệu thành corpus (danh sách các văn bản)
def create_corpus(data):
    return [content.replace("\n", " ").strip() for content in data.values()]

# Token hóa văn bản
def tokenize_corpus(corpus):
    return [word_tokenize(doc.lower()) for doc in corpus]

# Áp dụng BM25 để xếp hạng văn bản
def apply_bm25(corpus, query):
    tokenized_corpus = tokenize_corpus(corpus)
    tokenized_query = word_tokenize(query.lower())
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    return scores

# Sử dụng SBERT để rerank kết quả BM25
def rerank_with_sbert(query, top_k_docs):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    doc_embeddings = sbert_model.encode(top_k_docs, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    return cosine_scores.squeeze().cpu().numpy()

# Đánh giá độ chính xác
def evaluate_accuracy(corpus, training_data, top_k_bm25, top_k_rerank):
    correct = 0
    total = len(training_data)
    corpus_keys = list(articles.keys())

    for query, expected_keys in training_data.items():
        query = query.strip()  # Xóa khoảng trắng thừa
        bm25_scores = apply_bm25(corpus, query)  # Tính điểm BM25
        
        # Lấy top-k từ BM25
        top_k_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k_bm25]
        top_k_bm25_docs = [corpus[idx] for idx in top_k_bm25_idx]
        top_k_bm25_keys = [corpus_keys[idx] for idx in top_k_bm25_idx]
        
        # Sử dụng SBERT để rerank top-k BM25
        rerank_scores = rerank_with_sbert(query, top_k_bm25_docs)
        final_ranking_idx = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)[:top_k_rerank]
        final_top_keys = [top_k_bm25_keys[idx] for idx in final_ranking_idx]

        print(f"Query: {query}")
        print(f"Expected answers: {expected_keys}")
        print(f"Top-{top_k_rerank} Results after rerank: {final_top_keys}")

        # Kiểm tra nếu bất kỳ kết quả nào khớp với kết quả mong đợi
        if any(key in expected_keys for key in final_top_keys):
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Đọc dữ liệu
articles_path = "text/articlesFull.json"
training_data_path = "text/TrainingData(2).json"

articles = load_json_file(articles_path)
training_data = load_json_file(training_data_path)

if isinstance(articles, dict):
    corpus = create_corpus(articles)
else:
    print("Dữ liệu articles.json không đúng định dạng.")
    corpus = []

# Tính độ chính xác
if corpus and isinstance(training_data, dict):
    top_k_bm25 = 20  # Số lượng top-k từ BM25
    top_k_rerank = 3  # Số lượng top-k sau rerank
    accuracy = evaluate_accuracy(corpus, training_data, top_k_bm25, top_k_rerank)
    print(f"Độ chính xác (Top-{top_k_rerank} sau rerank): {accuracy * 100:.2f}%")
else:
    print("Dữ liệu không hợp lệ.")
