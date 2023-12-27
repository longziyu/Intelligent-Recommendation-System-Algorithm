import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
import numpy as np
import matplotlib.pyplot as plt

# 数据准备
class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = ratings['UserID'].values
        self.movies = ratings['MovieID'].values
        self.ratings = ratings['Rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

def load_data():
    ratings = pd.read_csv('path_to_movielens_1m/ratings.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
    # 将 UserID 和 MovieID 转换为连续整数索引
    ratings['UserID'] = ratings['UserID'].astype("category").cat.codes
    ratings['MovieID'] = ratings['MovieID'].astype("category").cat.codes
    return ratings

# 模型定义
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc = nn.Sequential(
            nn.Linear(embedding_size*2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input)
        x = torch.cat([user_embedded, movie_embedded], dim=-1)
        x = self.fc(x)
        return x.squeeze()

# 数据加载
ratings = load_data()
user_ids = ratings['UserID'].unique()
movie_ids = ratings['MovieID'].unique()
num_users, num_movies = len(user_ids), len(movie_ids)
embedding_size = 50

train_data, test_data = train_test_split(ratings, test_size=0.2)
train_dataset = MovieLensDataset(train_data)
test_dataset = MovieLensDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

writer = SummaryWriter('runs/movielens_experiment')
# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RecommenderNet(num_users, num_movies, embedding_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (users, movies, ratings) in enumerate(train_loader):
        users, movies, ratings = users.long().to(device), movies.long().to(device), ratings.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(users, movies)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 记录到 TensorBoard
        if batch_idx % 10 == 0:  # 每 10 个批次记录一次
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')
    writer.add_scalar('Average Training Loss per Epoch', average_loss, epoch)
# 关闭 SummaryWriter
writer.close()

'''
# 评分预测示例
def predict_rating(user_id, movie_id):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor([user_id]).to(device), torch.tensor([movie_id]).to(device))
    return prediction.item()

# 示例预测
sample_user, sample_movie = 1, 2
predicted_rating = predict_rating(sample_user, sample_movie)
print(f'Predicted Rating for user {sample_user} and movie {sample_movie}: {predicted_rating}')
'''

# 预测评分矩阵
def predict_matrix(model, num_users, num_movies):
    matrix = np.zeros((num_users, num_movies))
    model.eval()
    with torch.no_grad():
        for user_id in range(num_users):
            for movie_id in range(num_movies):
                prediction = model(torch.tensor([user_id]).long().to(device), torch.tensor([movie_id]).long().to(device))
                matrix[user_id, movie_id] = prediction.item()
    return matrix

# 生成预测矩阵
pred_matrix = predict_matrix(model, 50, 50)

# 绘制热图
plt.figure(figsize=(10, 8))
plt.imshow(pred_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('User-Movie Rating Predictions')
plt.xlabel('Movie ID')
plt.ylabel('User ID')
plt.show()

# 将预测矩阵保存到文件
np.savetxt('predicted_ratings_matrix.csv', pred_matrix, delimiter=',')

