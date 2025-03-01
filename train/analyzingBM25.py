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
    rank_distribution = {
                "0-10": 0,
                "11-30": 0,
                "31-50": 0,
                "51-100": 0,
                "101-200": 0,
                "201-500": 0,
                "Out of Top 500": 0
    }
    for query, expected_keys in training_data.items():
        scores = scores_per_query[query]
        top_k_idx = np.argsort(scores)[::-1]  # Lấy top-k nhanh bằng numpy
        top_k_keys = [corpus_keys[idx] for idx in top_k_idx]

        for key in expected_keys:
            rank = top_k_keys.index(key) + 1  # Xếp hạng bắt đầu từ 1
            # Phân loại rank vào nhóm phù hợp
            if rank <= 10:
                rank_distribution["0-10"] += 1
            elif rank <= 30:
                rank_distribution["11-30"] += 1
            elif rank <= 50:
                rank_distribution["31-50"] += 1
            elif rank <= 100:
                rank_distribution["51-100"] += 1
            elif rank <= 200:
                rank_distribution["101-200"] += 1
            elif rank <= 500:
                rank_distribution["201-500"] += 1
            else:
                rank_distribution["Out of Top 500"] += 1

            # In kết quả
    summ = sum(rank_distribution.values())
    print("📊 **Thống kê Expected Key theo vị trí trong bảng xếp hạng BM25:**")
    for key, count in rank_distribution.items():
        print(f"{key}: {count}:{count/summ}")

    label_set = set(expected_keys)
    total_queries.append((label_set, top_k_keys))

    overall_precision, overall_recall, overall_f2 = evaluate_F2_overall(total_queries)
    return overall_precision,overall_recall,overall_f2




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

    # top_k_list = [50]
    # for top_k in top_k_list :
    #     precision,recall,f2 = evaluate_recall(corpus, articles, training_data, bm25, tokenized_corpus, scores_per_query, top_k)
    #     print(f"Recall (Top-{top_k}): {recall * 100:.2f}%")
    #     print(precision)
    #     print(f2)
    #     recall = round(recall,4)
    #     top_.append(top_k)
    #     recall_.append(recall)

    # df = pd.DataFrame({"Top_k":top_,"Recall point":recall_})
    # df.to_csv("BM25_recall_result_with_BM25_shortlist", index=False)
else:
    print("Dữ liệu không hợp lệ.")


import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize

# Load dữ liệu mapping article -> chapter
true_chapters_path = "train/true_chapters.json"
with open(true_chapters_path, "r", encoding="utf-8") as file:
    article_to_chapter = json.load(file)

# Xây dựng ánh xạ chapter -> danh sách các articles
chapter_to_articles = defaultdict(list)
for article, chapter in article_to_chapter.items():
    chapter_to_articles[chapter].append(article)

def evaluate_chapter_coverage(articles, training_data, bm25, tokenized_corpus, scores_per_query):
    corpus_keys = list(articles.keys())  # Danh sách article IDs
    total_queries = 0
    matched_queries = 0
    article_counts = []

    for query, expected_keys in training_data.items():
        # Tính điểm BM25 cho truy vấn
        scores = scores_per_query[query]
        top_idx = np.argsort(scores)[-500:][::-1]  # Chọn 50 điều luật có điểm cao nhất
        top_articles = [corpus_keys[idx] for idx in top_idx]

        # **Đếm tần suất xuất hiện của chapters trong top 50 articles**
        chapter_counts = Counter(article_to_chapter.get(article, -1) for article in top_articles)
        
        # **Lấy 3 chapter có số lượng articles nhiều nhất**
        top_3_chapters = [chapter for chapter, _ in chapter_counts.most_common(22)]

        # **Lọc lại danh sách articles trong 3 chapters được chọn**
        selected_articles = []
        for ch in top_3_chapters:
            if ch in chapter_to_articles:
                articles_in_chapter = chapter_to_articles[ch]  # Danh sách articles của chương
                if len(articles_in_chapter) > 0:
                    # Nếu chương có ≥ 50 articles, lấy 50 article có điểm BM25 cao nhất
                    article_scores = {art: scores[corpus_keys.index(art)] for art in articles_in_chapter if art in corpus_keys}
                    top_50_articles = sorted(article_scores, key=article_scores.get, reverse=True)
                    selected_articles.extend(top_50_articles)
                else:
                    # Nếu chương có < 50 articles, lấy toàn bộ
                    selected_articles.extend(articles_in_chapter)

        article_counts.append(len(selected_articles))  # Ghi lại số lượng articles thực sự phải tính toán

        # **Kiểm tra xem expected_keys có nằm trong danh sách articles được chọn không**
        for expected_key in expected_keys:
            if expected_key in selected_articles:
                matched_queries += 1  # Nếu có ít nhất 1 expected key nằm trong danh sách lọc

        total_queries += len(expected_keys)

    # **Tính tỷ lệ expected_keys nằm trong tập articles sau khi lọc**
    match_ratio = matched_queries / total_queries if total_queries > 0 else 0
    print(f"Tỷ lệ expected_keys nằm trong 3 chương có số articles nhiều nhất: {match_ratio:.2%}")

    # **Tính toán thống kê**
    mean_articles = np.mean(article_counts)
    median_articles = np.median(article_counts)
    max_articles = np.max(article_counts)
    min_articles = np.min(article_counts)

    print(f"📊 Thống kê số articles thực tế cần tính toán:")
    print(f"Mean (trung bình): {mean_articles:.2f}")
    print(f"Median (trung vị): {median_articles}")
    print(f"Max (lớn nhất): {max_articles}")
    print(f"Min (nhỏ nhất): {min_articles}")

    return match_ratio, mean_articles, median_articles, max_articles, min_articles

# Chạy phân tích
# match_ratio, mean_articles, median_articles, max_articles, min_articles = evaluate_chapter_coverage(
#     articles, training_data, bm25, tokenized_corpus, scores_per_query
# )


import json
import numpy as np
import pandas as pd
from collections import defaultdict



def check_chapter_in_top_k(articles, training_data, bm25, tokenized_corpus, scores_per_query, top_k_values=[10, 20, 50, 100]):
    corpus_keys = list(articles.keys())  # Danh sách article IDs
    results = {k: {"total_queries": 0, "matched_queries": 0} for k in top_k_values}

    for query, expected_keys in training_data.items():
        # Tính điểm BM25 cho truy vấn
        scores = scores_per_query[query]
        sorted_indices = np.argsort(scores)[::-1]  # Sắp xếp theo điểm BM25 giảm dần
        
        # Lấy chương của expected keys
        expected_chapters = {article_to_chapter.get(key, -1) for key in expected_keys}

        for top_k in top_k_values:
            top_k_articles = [corpus_keys[idx] for idx in sorted_indices[:top_k]]
            top_k_chapters = {article_to_chapter.get(article, -1) for article in top_k_articles}

            # Kiểm tra xem có expected chapter nào nằm trong tập chapter của top-k articles không
            if any(ch in top_k_chapters for ch in expected_chapters):
                results[top_k]["matched_queries"] += 1

            results[top_k]["total_queries"] += 1

    # Tính tỷ lệ
    for top_k in top_k_values:
        total = results[top_k]["total_queries"]
        matched = results[top_k]["matched_queries"]
        match_ratio = matched / total if total > 0 else 0
        print(f"Tỷ lệ expected keys có chapter xuất hiện trong top-{top_k} articles: {match_ratio:.2%}")

    return results

# Chạy kiểm tra
results = check_chapter_in_top_k(articles, training_data, bm25, tokenized_corpus, scores_per_query)
