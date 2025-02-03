import json
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import os
print(os.listdir('/kaggle/input'))
# Tải bộ dữ liệu cần thiết cho NLTK
nltk.download('punkt')

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

# Đánh giá độ chính xác
def evaluate_accuracy(corpus, training_data, top_k=3):
    correct = 0
    total = len(training_data)
    corpus_keys = list(articles.keys())

    for query, expected_keys in training_data.items():
        query = query.strip()  # Xóa khoảng trắng thừa
        scores = apply_bm25(corpus, query)  # Tính điểm BM25
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_k_keys = [corpus_keys[idx] for idx in top_k_idx]
        
        print(f"Query: {query}")
        print (f"Expected answers: {expected_keys}")
        print(f"Top-{top_k} Results: {top_k_keys}")
        
        # Kiểm tra nếu bất kỳ kết quả nào khớp với kết quả mong đợi
        if any(key in expected_keys for key in top_k_keys):
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
    accuracy = evaluate_accuracy(corpus, training_data, top_k=3)
    print(f"Độ chính xác (Top-3): {accuracy * 100:.2f}%")
else:
    print("Dữ liệu không hợp lệ.")
