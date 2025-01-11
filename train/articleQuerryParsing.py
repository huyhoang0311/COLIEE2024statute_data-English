import json
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Tải bộ dữ liệu cần thiết cho NLTK
nltk.download('punkt_tab')

# Đọc nội dung từ file JSON
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Chuyển dữ liệu thành corpus (danh sách các văn bản)
def create_corpus(data):
    return [article.replace("\n", " ") for article in data]

# Token hóa văn bản
def tokenize_corpus(corpus):
    return [word_tokenize(doc.lower()) for doc in corpus]

# Áp dụng BM25 để xếp hạng văn bản
def apply_bm25(corpus, query):
    # Token hóa các văn bản và truy vấn
    tokenized_corpus = tokenize_corpus(corpus)
    tokenized_query = word_tokenize(query.lower())

    # Tạo đối tượng BM25 và tính toán điểm cho mỗi văn bản
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    
    return scores

# Đọc và xử lý file JSON
file_path = r'text/articles.json'  # Thay 'your_file.json' bằng đường dẫn file JSON của bạn
data = load_json_file(file_path)

# Kiểm tra dữ liệu đã được load chính xác
if isinstance(data, list):
    corpus = create_corpus(data)
else:
    print("Dữ liệu không phải là danh sách các bài viết.")
    corpus = []

# Truy vấn tìm kiếm
query = "A special provision that releases warranty can be made, but in that situation, when there are rights that the seller establishes on his/her own for a third party, the seller is not released of warranty."

# Áp dụng BM25 nếu corpus hợp lệ
if corpus:
    scores = apply_bm25(corpus, query)

    # Lấy top 3 bài viết có điểm số cao nhất
    top_3_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    # In kết quả top 3 bài viết
    for idx in top_3_idx:
        print(f"Articles {idx+1}: Score {scores[idx]}")
else:
    print("Không có văn bản để xử lý.")
