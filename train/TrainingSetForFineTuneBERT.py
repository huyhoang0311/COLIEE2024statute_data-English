import json
import pandas as pd
import random

# Đọc dữ liệu
with open('/kaggle/input/coliee/COLIEE2024statute_data-English/text/TrainingData(2).json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

with open('/kaggle/input/coliee/COLIEE2024statute_data-English/text/articlesFull.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

# Danh sách tất cả mã điều luật
all_law_ids = list(articles.keys())

triplets = []

# Tạo triplets (query, positive, negative)
for query, relevant_laws in training_data.items():
    positive_ids = relevant_laws
    negative_ids = list(set(all_law_ids) - set(positive_ids))

    for positive_id in positive_ids:
        # Chọn ngẫu nhiên 1 negative sample để đối lập với positive
        negative_id = random.choice(negative_ids)

        # Lưu triplet
        triplets.append({
            'query': query,
            'positive': articles.get(positive_id, ""),
            'negative': articles.get(negative_id, "")
        })

# Xuất ra CSV ở thư mục Kaggle Working Directory
df = pd.DataFrame(triplets)
output_path = "/kaggle/working/train_triplets.csv"
df.to_csv(output_path, index=False)

print(f"Đã tạo {len(triplets)} cặp triplets và lưu vào {output_path}")
