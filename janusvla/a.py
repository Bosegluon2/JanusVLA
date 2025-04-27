from transformers import T5Tokenizer, T5Model
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. 特征提取模块（基于你提供的代码改进）
class TextFeatureExtractor:
    def __init__(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5Model.from_pretrained(model_path)
        self.model.eval()  # 固定模型参数
        
    def extract_features(self, texts, batch_size=16):
        features = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                encoder_outputs = self.model.encoder(**inputs)
                last_hidden_states = encoder_outputs.last_hidden_state
                batch_features = last_hidden_states.mean(dim=1)
                
            features.append(batch_features)
            
        return torch.cat(features, dim=0)

# 2. 数据集定义
class ReviewDataset(Dataset):
    def __init__(self, features, scores):
        self.features = features
        self.scores = scores
        
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'score': torch.tensor(self.scores[idx], dtype=torch.float32)
        }

# 3. 回归模型
class RatingRegressor(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.regressor(x)

# 4. 训练流程
def train_model(train_loader, val_loader, epochs=20):
    
    model = RatingRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            features = batch['features'].to(device)
            scores = batch['score'].to(device)
            
            outputs = model(features).squeeze()
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                scores = batch['score'].to(device)
                outputs = model(features).squeeze()
                val_loss += criterion(outputs, scores).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

# 主流程
if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv('Reviews.csv')  # 替换为你的数据路径
    
    # 提取特征
    extractor = TextFeatureExtractor("./t5-v1_1-base/")
    features = extractor.extract_features(df['Text'].tolist())
    
    # 转换为numpy
    X = features.numpy()
    y = df['Score'].values.astype(np.float32)
    
    # 数据分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建DataLoader
    train_dataset = ReviewDataset(torch.tensor(X_train), y_train)
    val_dataset = ReviewDataset(torch.tensor(X_val), y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 训练模型
    trained_model = train_model(train_loader, val_loader)
    
    # 评估最终模型
    trained_model.load_state_dict(torch.load('best_model.pth'))
    trained_model.eval()
    
    with torch.no_grad():
        val_preds = trained_model(torch.tensor(X_val).to(device)).cpu().numpy().squeeze()
    
    print(f'MSE: {mean_squared_error(y_val, val_preds):.4f}')
    print(f'MAE: {mean_absolute_error(y_val, val_preds):.4f}')
