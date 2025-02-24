import json
import nltk

# Tải bộ dữ liệu cần thiết cho NLTK
nltk.download('punkt')

# Đọc nội dung từ file JSON
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# File JSON chứa dữ liệu
data_path = 'train/test.json'

# Đọc dữ liệu
Data = load_json_file(data_path)

# Biến đếm tổng số query
cnt = 0

# Mảng lưu số lượng query có n expected keys (giả sử tối đa 10)
cnt_expected_key = [0] * 11  

# Lặp qua từng query trong Data
for query, expected in Data.items():
    cnt += 1  # Đếm số lượng query
    cntt = len(expected)  # Lấy số lượng expected keys trong query này

    # Chỉ cập nhật nếu trong phạm vi hợp lệ
    if cntt < len(cnt_expected_key):
        cnt_expected_key[cntt] += 1  

# In tổng số query
print("Tổng số query:", cnt)

# In số lượng query theo từng mức expected keys
for i, key_count in enumerate(cnt_expected_key):
    print(f"Số query có {i} expected keys: {key_count}")
