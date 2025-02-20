import json
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

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

# Hàm lấy top_k văn bản liên quan nhất cho mỗi query
def get_top_k_queries(corpus_keys, training_data, scores_per_query, top_k):
    top_k_queries = {}
    for query in training_data.keys():
        scores = scores_per_query[query]
        top_k_idx = np.argsort(scores)[-top_k:][::-1]
        top_k_keys = [corpus_keys[idx] for idx in top_k_idx]
        top_k_queries[query] = top_k_keys
    return top_k_queries

# Đường dẫn tới các file dữ liệu
articles_path = "text/articlesFull.json"
training_data_path = "text/TrainingData(2).json"

articles = load_json_file(articles_path)
training_data = load_json_file(training_data_path)

if isinstance(articles, dict):
    corpus = create_corpus(articles)
else:
    print("Dữ liệu articles.json không đúng định dạng.")
    corpus = []

corpus_keys = list(articles.keys())
tokenized_corpus = tokenize_corpus(corpus)

# Thiết lập BM25
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# Tính điểm BM25 trước cho tất cả các query
scores_per_query = {
    query: bm25.get_scores(word_tokenize(query.lower()))
    for query in training_data.keys()
}

# Lấy top_k văn bản liên quan nhất cho mỗi query
top_k = 30
top_k_queries = get_top_k_queries(corpus_keys, training_data, scores_per_query, top_k)

triplets = []

# Tạo triplets (query, positive, negative)
for query, positive_ids in training_data.items():
    top_k_query = top_k_queries[query]
    # Loại bỏ positive khỏi negative
    negative_ids = list(set(top_k_query)-set(positive_ids))

    for positive_id in positive_ids:
        for negative_id in negative_ids:
            triplets.append({
                'query': query,
                'positive': articles.get(positive_id, ""),
                'negative': articles.get(negative_id, "")
            })

# Xuất ra CSV
df = pd.DataFrame(triplets)
output_path = "/kaggle/working/train_triplets.csv"
df.to_csv(output_path, index=False)

print(f"Đã tạo {len(triplets)} cặp triplets và lưu vào {output_path}")
