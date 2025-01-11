from bs4 import BeautifulSoup 
import pandas as pd
import json 
import re
import os

pairs = []
def solve_xml(file_path) :
   #file_path = "/kaggle/input/datacoliee/ColieeData/train/riteval_H18_en.xml"

    with open (file_path,'r') as f :
        data = f.read()

    # passing data inside the beautifulsoup parser
    bs_data = BeautifulSoup(data, "xml")
    bs_t1 = bs_data.find_all('t1')
    bs_t2 = bs_data.find_all('t2')
    #print(bs_t1)
    values = []
    for text in bs_t1 :
        values.append(text.get_text())
   
    keys = []
    for text in bs_t2 :
        keys.append(text.get_text())
   
    
    for key,value in zip(keys,values):
        lines = value.splitlines()  # Tách đoạn văn bản thành danh sách các dòng

        # Lọc các dòng bắt đầu bằng "Article" và trích xuất số thứ tự
        article_numbers = [
               re.search(r"^Article\s*(\d+(?:-\d+)?)", line).group(1)  # Trích xuất số, hỗ trợ cả số dạng "456-2"
               for line in lines
               if line.strip().startswith("Article") and re.search(r"^Article\s*(\d+(?:-\d+)?)", line)
        ]
        pairs.append((key, article_numbers))
    return pairs
    
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H18_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H19_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H20_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H21_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H22_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H23_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H24_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H25_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H26_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H27_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H28_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H29_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_H30_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_R01_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_R02_en.xml")
solve_xml("/kaggle/input/datacoliee/ColieeData/train/riteval_R03_en.xml")

result_dict = {key: article_numbers for key, article_numbers in pairs}

file_path = "/kaggle/working/output.json"
with open(file_path, 'w') as f:
    json.dump([result_dict], f, indent=4)  # Ghi dữ liệu vào dưới dạng danh sách

print(f"Đã thêm dữ liệu vào {file_path}")
