import json
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