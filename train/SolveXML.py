from bs4 import BeautifulSoup 


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
    pairs = []
    
    for key,value in zip(keys,values):
        lines = value.splitlines()  # Tách đoạn văn bản thành danh sách các dòng
        article_lines = [line for line in lines if line.strip().startswith("Article")]  # Lọc các dòng bắt đầu bằng "Article"
        pairs.append((key,article_lines))
    return pairs 
    
file_path = "/kaggle/input/datacoliee/ColieeData/train/riteval_H18_en.xml"
solve_xml(file_path)
