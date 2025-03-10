from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os
os.environ["WANDB_DISABLED"] = "true"

# Load mô hình LegalBERT để fine-tune
#model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
model = SentenceTransformer('msmacro-base-dot-v5',device='cuda')

# Đọc tập train_triplets.csv vừa tạo
df = pd.read_csv("/kaggle/input/coliee-update-2024/COLIEE2024statute_data-English/text/train_triplets.csv")

# Chuyển thành dạng InputExample để huấn luyện
train_samples = []
for _, row in df.iterrows():
    train_samples.append(
        InputExample(
            texts=[row['query'], row['positive'], row['negative']]
        )
    )

# DataLoader để load batch
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)

# Loss phù hợp cho search/rerank
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tuning (đào tạo)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='/kaggle/working/fine_tuned_legalbert'
)

print("Huấn luyện xong. Model đã lưu tại: /kaggle/working/fine_tuned_legalbert")
