
import re
import json
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier


def split_into_articles_from_file(file_path):
    # Đọc nội dung từ file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Loại bỏ các dòng bắt đầu bằng "Chapter" và "Section"
    text = '\n'.join([line for line in text.split('\n') if not (line.startswith('Chapter') or line.startswith('Section'))])
    print(text)
    # Loại bỏ các dòng bắt đầu bằng "(" và sau đó là chữ
    text = '\n'.join([line for line in text.split('\n') if not re.match(r'^\(\s*[a-zA-Z]', line.lstrip())])
    
    # Sử dụng regex để tách các điều khoản
    articles = re.findall(r'(Article \d[\-\d]*\s+.*?)(?=\nArticle \d|\Z)', text, re.DOTALL)
    return articles

def predict_with_tree_and_similarity(query):
    query_vector = vectorizer.transform(query)  # Mã hóa query mới

    # Sử dụng Decision Tree để dự đoán
    predicted_label = clf.predict(query_vector)

    return predicted_label

json_path = r'text\articles.json'

# Khởi tạo lemmatizer
lemmatizer = WordNetLemmatizer()
with open(json_path, 'r') as file:
    temp = json.load(file)
corpus = [item for item in temp]
# Mảng lưu các nhãn (số)
labels = []

# Duyệt qua từng chuỗi trong data và trích xuất số ngay sau từ "Article"
for idx, article in enumerate(corpus):
    # Tìm số ngay sau từ "Article"
    match = re.search(r'Article (\S+)', article)
    if match:
        # Nếu tìm thấy, lấy số đầu tiên và chuyển thành kiểu int
        labels.append((match.group(1)))
    # Loại bỏ tất cả các số từ văn bản
    corpus[idx] = re.sub(r'\b\d+\b', '', article)


processed_corpus = []
for doc in corpus:
    lemmatized_text = " ".join([lemmatizer.lemmatize(word, pos='v') for word in doc.split()])
    processed_corpus.append(lemmatized_text)

#Stop word:
custom_stopwords = list(set(ENGLISH_STOP_WORDS).union(set(string.digits)))
vectorizer = TfidfVectorizer(stop_words = custom_stopwords)
article_vectors = vectorizer.fit_transform(processed_corpus)


# Huấn luyện RFC trên các vector của các bài viết
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(article_vectors, labels)



queries = [
    "A special provision that releases warranty can be made, but in that situation, when there are rights that the seller establishes on his/her own for a third party, the seller is not released of warranty."
]
print(type(queries))
# Tien xu ly query
processed_queries = []
for doc in queries:
    lemmatized_text = " ".join([lemmatizer.lemmatize(word, pos='v') for word in doc.split()])
    processed_queries.append(lemmatized_text)
result = predict_with_tree_and_similarity(processed_queries)

print(result)