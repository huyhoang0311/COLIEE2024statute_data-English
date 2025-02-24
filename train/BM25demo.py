import json
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import csv
import pandas as pd
# Tải bộ dữ liệu cần thiết cho NLTK
nltk.download('punkt_tab')

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

# Đánh giá F2-score
def evaluate_F2_single(label_set, predict_set):
    correct_retrieved = len(label_set.intersection(predict_set))
    precision = correct_retrieved / len(predict_set) if len(predict_set) > 0 else 0
    recall = correct_retrieved / len(label_set) if len(label_set) > 0 else 0
    if precision + recall == 0:
        f2_measure = 0
    else:
        f2_measure = (5 * precision * recall) / (4 * precision + recall)
    return precision, recall, f2_measure

def evaluate_F2_overall(queries):
    total_precision = 0
    total_recall = 0
    num_queries = len(queries)
    for label_set, predict_set in queries:
        precision, recall, _ = evaluate_F2_single(label_set, predict_set)
        total_precision += precision
        total_recall += recall
    avg_precision = total_precision / num_queries if num_queries > 0 else 0
    avg_recall = total_recall / num_queries if num_queries > 0 else 0
    if avg_precision + avg_recall == 0:
        overall_f2 = 0
    else:
        overall_f2 = (5 * avg_precision * avg_recall) / (4 * avg_precision + avg_recall)
    return avg_precision, avg_recall, overall_f2

# Đánh giá Recall theo top_k
def evaluate_recall(corpus, articles, training_data, bm25, tokenized_corpus, scores_per_query, top_k):
    corpus_keys = list(articles.keys())
    total_queries = []

    for query, expected_keys in training_data.items():
        scores = scores_per_query[query]
        top_k_idx = np.argsort(scores)[-top_k:][::-1]  # Lấy top-k nhanh bằng numpy
        top_k_keys = [corpus_keys[idx] for idx in top_k_idx]
        print(query)
        print(top_k_keys[0])
        for key in expected_keys:
            if key in top_k_keys:
                index = top_k_keys.index(key)
                score = scores[top_k_idx[index]]
                
                print(f"Key: {key}, Final Rerank Score: {score:.4f}, Rank: {index}")
            else:
                print(f"Key: {key} không có trong danh sách kết quả")              



        label_set = set(expected_keys)
        total_queries.append((label_set, top_k_keys))

    overall_precision, overall_recall, overall_f2 = evaluate_F2_overall(total_queries)
    return overall_precision,overall_recall,overall_f2




# Đọc dữ liệu
#articles_path = "/kaggle/input/coliee/COLIEE2024statute_data-English/text/articlesFull.json"
#training_data_path = "/kaggle/input/coliee/COLIEE2024statute_data-English/text/TrainingData(2).json"

articles_path = "text/articlesFull.json"
training_data_path = "train/test.json"



articles = load_json_file(articles_path)
training_data = load_json_file(training_data_path)

if isinstance(articles, dict):
    corpus = create_corpus(articles)
else:
    print("Dữ liệu articles.json không đúng định dạng.")
    corpus = []

# Tính độ chính xác
if corpus and isinstance(training_data, dict):
    tokenized_corpus = tokenize_corpus(corpus)

    #điều chỉnh tham số bm25
    #k1 = 1.55
    #b = 0.8
    bm25 = BM25Okapi(tokenized_corpus)

    # Tính trước điểm số cho mỗi query để không phải chạy lại nhiều lần
    scores_per_query = {
        query: bm25.get_scores(word_tokenize(query.lower()))
        for query in training_data.keys()
    }
    top_ = []
    recall_ = []

    # for top_k in range(1, 501):
    #     recall = evaluate_recall(corpus, articles, training_data, bm25, tokenized_corpus, scores_per_query, top_k)
    #     print(f"Recall (Top-{top_k}): {recall * 100:.2f}%")
    #     recall = round(recall,4)
    #     top_.append(top_k)
    #     recall_.append(recall)
    #df = pd.DataFrame({"Top_k":top_,"Recall point":recall_})
    #df.to_csv("BM25_recall_result",index = False)   

    top_k_list = [700]
    for top_k in top_k_list :
        precision,recall,f2 = evaluate_recall(corpus, articles, training_data, bm25, tokenized_corpus, scores_per_query, top_k)
        print(f"Recall (Top-{top_k}): {recall * 100:.2f}%")
        print(precision)
        print(f2)
        recall = round(recall,4)
        top_.append(top_k)
        recall_.append(recall)

    #df = pd.DataFrame({"Top_k":top_,"Recall point":recall_})
    #df.to_csv("BM25_recall_result_with_BM25", index=False)
else:
    print("Dữ liệu không hợp lệ.")
