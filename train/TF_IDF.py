 
import re
import json
import string
import numpy as np
import sys
sys.path.append('/kaggle/input/f2-score')
import F2_measure
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics.pairwise import cosine_similarity

def split_into_articles_from_file(file_path):
    # Đọc nội dung từ file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Loại bỏ các dòng bắt đầu bằng "Chapter" và "Section"
    text = '\n'.join([line for line in text.split('\n') if not (line.startswith('Chapter') or line.startswith('Section'))])
    # Loại bỏ các dòng bắt đầu bằng "(" và sau đó là chữ
    text = '\n'.join([line for line in text.split('\n') if not re.match(r'^\(\s*[a-zA-Z]', line.lstrip())])
    
    # Sử dụng regex để tách các điều khoản
    articles = re.findall(r'(Article \d[\-\d]*\s+.*?)(?=\nArticle \d|\Z)', text, re.DOTALL)
    return articles

def predict_with_tree_and_similarity(queries):
    queries_vector = vectorizer.transform(queries)  # Mã hóa query mới
    queries = queries_vector.toarray()
    #Sử dụng Decision Tree để dự đoán
    predicted_label_proba = clf.predict_proba(queries)

    return predicted_label_proba

articles_path = '/kaggle/input/civil-json/civil_code_en-1to724-2.json'
traning_path = '/kaggle/input/hanghoi/TrainingData.json'
test_path = '/kaggle/input/f2-score/test.json'

# Khởi tạo lemmatizer
lemmatizer = WordNetLemmatizer()
# Mảng lưu các nhãn (số)
labels = []
classes = []
queries = []
label_set = []

with open(articles_path, 'r') as file:
    temp_articles = json.load(file)
corpus = [item for item in temp_articles]
# Duyệt qua từng chuỗi trong data và trích xuất số ngay sau từ "Article"
for idx, article in enumerate(corpus):
    # Tìm số ngay sau từ "Article"
    match = re.search(r'Article (\S+)', article)
    if match:
        # Nếu tìm thấy, lấy số đầu tiên
        labels.append((match.group(1)))
        classes.append((match.group(1)))
    # Loại bỏ tất cả các số từ văn bản
    corpus[idx] = re.sub(r'\b\d+\b', '', article)

classes = np.array(classes)

with open(traning_path, 'r') as training_file:
    temp_trainingData = json.load(training_file)
    for item in temp_trainingData[0]:
        corpus.append(item)
    

processed_corpus = []
for doc in corpus:
    lemmatized_text = " ".join([lemmatizer.lemmatize(word, pos='v') for word in doc.split()])
    processed_corpus.append(lemmatized_text)
    

#Multi-label solution:
mlb = MultiLabelBinarizer()
for i, label in enumerate(labels):
    labels[i] = [label]
for item in temp_trainingData[0]:
        labels.append(temp_trainingData[0][item])
label_bin = mlb.fit_transform(labels)

#Stop word:
custom_stopwords = list(set(ENGLISH_STOP_WORDS).union(set(string.digits)))
vectorizer = TfidfVectorizer(stop_words = custom_stopwords)
article_vectors = vectorizer.fit_transform(processed_corpus)

# Huấn luyện RFC trên các vector của các bài viết
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(article_vectors, label_bin)

def get_test_data(test_path):
    with open(test_path, 'r') as testing_file:
        test_Data = json.load(testing_file)
    for item in test_Data[0]:
        queries.append(item)
        label_set.append(test_Data[0][item])

get_test_data(test_path)
# Tien xu ly query
processed_queries = []
for doc in queries:
    lemmatized_text = " ".join([lemmatizer.lemmatize(word, pos='v') for word in doc.split()])
    processed_queries.append(lemmatized_text)


proba = predict_with_tree_and_similarity(processed_queries)
result = []
for i, row in enumerate(proba):
    top_indices = np.argsort(row)[-3:]  # Chỉ số của top 3 xác suất
    top_classes = classes[top_indices]# Nhãn tương ứng
    result.append((set(label_set[i]), set(top_classes)))

print(result[0])

    
