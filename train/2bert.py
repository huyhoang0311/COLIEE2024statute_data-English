from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np
import pandas as pd
import json
import heapq

nltk.download('punkt_tab')

#Load mô hình 


#Xử lý file JSON
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
######################################
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
    total_f2 = 0
    num_queries = len(queries)
    for label_set, predict_set in queries:
        precision, recall, f2 = evaluate_F2_single(label_set, predict_set)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
    avg_precision = total_precision / num_queries if num_queries > 0 else 0
    avg_recall = total_recall / num_queries if num_queries > 0 else 0
    if avg_precision + avg_recall == 0:
        overall_f2 = 0
    else:
        #overall_f2 = (5 * avg_precision * avg_recall) / (4 * avg_precision + avg_recall)
        overall_f2 = total_f2 / num_queries
    return avg_precision, avg_recall, overall_f2
###########################################

def get_top_by_threshold(scores, corpus_keys, threshold):

    filtered_idxs = [i for i, score in enumerate(scores) if score >= threshold]
    # Nếu không có phần tử nào đạt threshold, lấy phần tử có điểm cao nhất
    if not filtered_idxs:
        max_score = max(scores)
        filtered_idxs = [i for i, score in enumerate(scores) if score == max_score]

    return filtered_idxs


def final_score(alpha,beta, score_per_query, bert_score_per_query,s2bert_score_per_query):
    final_scores = {}

    for query in score_per_query.keys():
        bm25_scores = np.array(score_per_query[query])
        sbert_scores = np.array(bert_score_per_query[query])
        s2bert_scores = np.array(s2bert_score_per_query[query])
        final_scores[query] = bm25_scores*alpha + (beta)*sbert_scores+(1-alpha-beta)*s2bert_scores
            
    return final_scores
###########################################
# Đánh giá độ chính xác
def evaluate_F2_score(corpus,article ,training_data,final_score_per_query, threshold):
    total_queries = []
    corpus_keys = list(article.keys())

    for query, expected_keys in training_data.items():

        final_scores = final_score_per_query[query]
        
      
        final_ranking_idx = get_top_by_threshold(final_scores,corpus_keys,threshold)

        final_top_keys = [corpus_keys[idx] for idx in final_ranking_idx]
        
        
        label_set = set(expected_keys)
        total_queries.append((label_set, final_top_keys))

    precision,recall, f2 = evaluate_F2_overall(total_queries)
    return precision,recall,f2

# Đọc dữ liệu
articles_path = "/kaggle/input/coliee-with-finetunebert/COLIEE2024statute_data-English/text/articlesFull.json"
training_data_path = "/kaggle/input/coliee-with-finetunebert/COLIEE2024statute_data-English/train/validation.json"
#training_data_path = "/kaggle/input/coliee-with-finetunebert/COLIEE2024statute_data-English/train/test.json"

######################################
articles = load_json_file(articles_path)
training_data = load_json_file(training_data_path)

if isinstance(articles, dict):
    corpus = create_corpus(articles)
else:
    print("Dữ liệu articles.json không đúng định dạng.")
    corpus = []
######################################


########################################

def min_max_normalization(score_per_query):
    normalize_score ={}
    for query, point in score_per_query.items():
        max_in_point = max(point)
        min_in_point = min(point)
        normalize_score[query] = [(p - min_in_point) / (max_in_point - min_in_point) for p in point]
    return normalize_score


#######################################
# Tính độ chính xác
if corpus and isinstance(training_data, dict):
    top_k_bm25 = len(corpus)  # Số lượng top-k từ BM25
    tokenized_corpus = tokenize_corpus(corpus)
    #bm25 = BM25Okapi(tokenized_corpus)



    file_path_1 = "path/to/your/file.csv"  # Thay thế bằng đường dẫn file thực tế
    file_path_2 = ""
    file_path_3 = ""
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    df3 = pd.read_csv(file_path_3)

    # Tạo dictionary
    scores_per_query = {
        row[0].strip(): row[1:].tolist()  # Key: Câu văn bản, Value: Mảng số
        for row in df1.itertuples(index=False)
    }

    bert_scores_per_query = {
        row[0].strip(): row[1:].tolist()  # Key: Câu văn bản, Value: Mảng số
        for row in df2.itertuples(index=False)
    }
    s2bert_scores_per_query = {
        row[0].strip(): row[1:].tolist()  # Key: Câu văn bản, Value: Mảng số
        for row in df3.itertuples(index=False)
    }

    
    scores_per_query = min_max_normalization(scores_per_query)
    bert_scores_per_query = min_max_normalization(bert_scores_per_query)
    s2bert_scores_per_query = min_max_normalization(s2bert_scores_per_query)
    
    corpus_keys = list(articles.keys())    
 

    alphas = np.arange(0, 1.05, 0.05)  # Bao gồm cả 1.0
    thresholds = np.arange(0, 1.05, 0.05)  
    betas = np.arange(0,1.05,0.05)
    
    
    results = []
    f2_max = 0
    alpha_best = 0
    threshold_best = 0
    beta_best = 0
    for beta in betas :
        for alpha in alphas:
            if alpha + beta <= 1 :
                final_scores_per_query = min_max_normalization(final_score(alpha,beta, scores_per_query, bert_scores_per_query,s2bert_scores_per_query))
            
                for x in thresholds:
                    # Đánh giá trên tập validation
                    precision, recall, f2 = evaluate_F2_score(
                        corpus, articles, training_data, final_scores_per_query,
                        threshold=x
                    )
            
                    # Lưu kết quả vào danh sách
                    results.append({
                        "Beta score" : beta,
                        "Alpha Score": alpha,
                        "Threshold": x,
                        "Precision": precision,
                        "Recall": recall,
                        "F2-score": f2
                    })
            
                    # Cập nhật giá trị tốt nhất
                    if f2 > f2_max:
                        f2_max = f2
                        alpha_best = alpha
                        threshold_best = x
                        beta_best = beta
            
                    print(f"Beta:{beta},Alpha: {alpha:.2f}, Threshold: {x:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F2-score: {f2:.4f}")
        
    # 📝 Tạo DataFrame sau khi vòng lặp kết thúc
    df_results = pd.DataFrame(results, columns=["Beta","Alpha Score", "Threshold", "Precision", "Recall", "F2-score"])   
    
    # 📂 Lưu kết quả vào CSV
    output_path = "/kaggle/working/beta_alpha_threshold_gridsearch.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\n✅ Kết quả tốt nhất: Beta = {beta_best},Alpha = {alpha_best}, Threshold = {threshold_best}, F2-score = {f2_max:.4f}")
    print(f"📂 File kết quả đã lưu tại: {output_path}")
else:
    print("Dữ liệu không hợp lệ.")