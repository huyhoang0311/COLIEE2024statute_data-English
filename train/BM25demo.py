import json
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
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

# Áp dụng BM25 để xếp hạng văn bản
def apply_bm25(corpus, query):
    tokenized_corpus = tokenize_corpus(corpus)
    tokenized_query = word_tokenize(query.lower())
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    return scores


#sử dụng F2_score để tính
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

# Đánh giá độ chính xác
def evaluate_accuracy(corpus, article, training_data, top_k):
    corpus_keys = list(article.keys())
    total_queries = []
    for query, expected_keys in training_data.items():
        query = query.strip()  # Xóa khoảng trắng thừa
        scores = apply_bm25(corpus, query)  # Tính điểm BM25
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_k_keys = [corpus_keys[idx] for idx in top_k_idx]
        
        #print(f"Query: {query}")
        #print (f"Expected answers: {expected_keys}")
        #print(f"Top-{top_k} Results: {top_k_keys}")
        
        label_set = set(expected_keys)
        total_queries.append((label_set, top_k_keys))
     
    _,overall_recall,_= evaluate_F2_overall(total_queries)
    print(f"Điểm recall tổng thể: {overall_recall:.4f}")
    return overall_recall

# Đọc dữ liệu
#articles_path = "/kaggle/input/coliee/COLIEE2024statute_data-English/text/articlesFull.json"
#training_data_path = "/kaggle/input/coliee/COLIEE2024statute_data-English/text/TrainingData(2).json"

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
    for i in range (1,201) :
        top_k = i
        accuracy = evaluate_accuracy(corpus, articles, training_data, top_k)
        print(f"Độ chính xác (Top- :{top_k} ): {accuracy * 100:.2f}%")
else:
    print("Dữ liệu không hợp lệ.")
