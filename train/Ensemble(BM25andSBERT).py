from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np
import pandas as pd
import json


nltk.download('punkt_tab')

#sbert_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
sbert_model = SentenceTransformer('/kaggle/input/coliee-with-finetunebert/COLIEE2024statute_data-English/fine_tuned_legalbert/kaggle/working/fine_tuned_legalbert', device='cuda')
#sbert_model = SentenceTransformer('nlpaueb/legal-bert-base-uncased', device='cuda')
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


# Sử dụng SBERT để rerank kết quả BM25
def rerank_with_sbert(query):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    doc_embeddings = corpus_embedding
    cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    return cosine_scores.squeeze().cpu().numpy()

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



def get_top_by_threshold(scores, corpus_keys, threshold):
    filtered_idxs = [i for i, score in enumerate(scores) if score >= threshold]
    max_score = max(scores)
    if not filtered_idxs : filtered_idxs = [i for i, score in enumerate(scores) if score == max_score]
    return filtered_idxs



def final_score(alpha, score_per_query, bert_score_per_query):
    final_scores = {}

    for query in score_per_query.keys():
        bm25_scores = np.array(score_per_query[query])
        sbert_scores = np.array(bert_score_per_query[query])
        final_scores[query] = bm25_scores*alpha + (1-alpha)*sbert_scores
            
    return final_scores


# Đánh giá độ chính xác
def evaluate_F2_score(corpus,article ,training_data,scores_per_query,bert_score_per_query,final_score_per_query ,top_k_bm25, top_k_rerank, threshold):
    total_queries = []
    corpus_keys = list(article.keys())

    for query, expected_keys in training_data.items():
        #Lấy điểm sẵn
        scores = scores_per_query[query]
        rerank_score = bert_score_per_query[query]
        final_scores = final_score_per_query[query]
        
        # Lấy top-k từ BM25
        top_k_bm25_idx = np.argsort(scores)[-top_k_bm25:][::-1]
       # top_k_bm25_idx = get_top_by_threshold(scores,corpus_keys,0.8)
        top_k_bm25_docs = [corpus[idx] for idx in top_k_bm25_idx]
        top_k_bm25_keys = [corpus_keys[idx] for idx in top_k_bm25_idx]

    

        
        
        # Sử dụng SBERT để rerank top-k BM25
        #rerank_scores = rerank_with_sbert(query, top_k_bm25_docs)
        #final_ranking_idx = np.argsort(rerank_score)[-top_k_rerank:][::-1]
        final_ranking_idx = get_top_by_threshold(final_scores,corpus_keys,threshold)
        #if len(final_ranking_idx) >= 2: final_ranking_idx = np.argsort(rerank_scores)[-top_k_rerank:][::-1]
      
      
        final_top_keys = [corpus_keys[idx] for idx in final_ranking_idx]
        
        
        #print(f"Query: {query}")
        #print(f"Expected answers: {expected_keys}")
        #print(f"Top-{top_k_rerank} Results after rerank: {final_top_keys}")

        # Kiểm tra nếu bất kỳ kết quả nào khớp với kết quả mong đợi
        
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
if corpus:
    corpus_embedding = sbert_model.encode(corpus,convert_to_tensor=True)

# Sử dụng SBERT để rerank kết quả BM25
def rerank_with_sbert(query):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    doc_embeddings = corpus_embedding
    cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    return cosine_scores.squeeze().cpu().numpy()


########################################

def min_max_normalization(score_per_query):
    normalize_score ={}
    for query, point in score_per_query.items():
        max_in_point = max(point)
        min_in_point = min(point)
        normalize_score[query] = [(p - min_in_point) / (max_in_point - min_in_point) for p in point]
    return normalize_score
    

# Tính độ chính xác
if corpus and isinstance(training_data, dict):
    top_k_bm25 = len(corpus)  # Số lượng top-k từ BM25
    top_k_rerank = 1 # Số lượng top-k sau rerank
    tokenized_corpus = tokenize_corpus(corpus)
    bm25 = BM25Okapi(tokenized_corpus)


    # Tính trước điểm số cho mỗi query để không phải chạy lại nhiều lần
    scores_per_query = {
        query: bm25.get_scores(word_tokenize(query.lower()))
        for query in training_data.keys()
    }
    bert_scores_per_query = {
        query : rerank_with_sbert(query)
        for query in training_data.keys()
    }
    scores_per_query = min_max_normalization(scores_per_query)
    bert_scores_per_query = min_max_normalization(bert_scores_per_query)
    
    corpus_keys = list(articles.keys())    
    #################################
    #df1 = pd.DataFrame.from_dict(scores_per_query,orient='index')
    #df1.columns = [f"{corpus_keys[idx]}" for idx in range (len(corpus_keys))]
    #df2 = pd.DataFrame.from_dict(bert_scores_per_query,orient='index')
    #df2.columns = [f"{corpus_keys[idx]}" for idx in range (len(corpus_keys))]
    #output_path_1 = "/kaggle/working/bm25point.csv"
    #output_path_2 = "/kaggle/working/bertpoint.csv"
    #df1.to_csv(output_path_1, index=True)
    #df2.to_csv(output_path_2,index=True)
    ################################

    #precision,recall,accuracy = evaluate_F2_score(corpus, articles, training_data, scores_per_query,bert_scores_per_query,final_scores_per_query, top_k_bm25, top_k_rerank)
    #print(f"Độ chính xác (Top-{top_k_rerank} sau rerank): {accuracy * 100:.2f}%")
    #print(precision)
    #print(recall)

    alphas = np.arange(0, 1.01, 0.01)  # Bao gồm cả 1.0
    thresholds = np.arange(0, 1.01, 0.01)  
    
    results = []
    f2_max = 0
    alpha_best = 0
    threshold_best = 0
    
    for alpha in alphas:
        final_scores_per_query = min_max_normalization(final_score(alpha, scores_per_query, bert_scores_per_query))
    
        for x in thresholds:
            # Đánh giá trên tập validation
            precision, recall, f2 = evaluate_F2_score(
                corpus, articles, training_data, scores_per_query, bert_scores_per_query, final_scores_per_query,
                top_k_bm25, top_k_rerank, threshold=x
            )
    
            # Lưu kết quả vào danh sách
            results.append({
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
    
            print(f"Alpha: {alpha:.2f}, Threshold: {x:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F2-score: {f2:.4f}")
    
    # 📝 Tạo DataFrame sau khi vòng lặp kết thúc
    df_results = pd.DataFrame(results, columns=["Alpha Score", "Threshold", "Precision", "Recall", "F2-score"])   
    
    # 📂 Lưu kết quả vào CSV
    output_path = "/kaggle/working/alpha_threshold_gridsearch.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\n✅ Kết quả tốt nhất: Alpha = {alpha_best}, Threshold = {threshold_best}, F2-score = {f2_max:.4f}")
    print(f"📂 File kết quả đã lưu tại: {output_path}")
    
else:
    print("Dữ liệu không hợp lệ.")