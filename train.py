import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 数据准备
class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users, self.movies, self.ratings = ratings['UserID'], ratings['MovieID'], ratings['Rating']

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

def load_data():
    ratings = pd.read_csv('path_to_movielens_1m/ratings.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
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

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RecommenderNet(num_users, num_movies, embedding_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for users, movies, ratings in train_loader:
        users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
        optimizer.zero_grad()
        outputs = model(users, movies)
        loss = criterion(outputs, ratings.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评分预测
def predict_rating(user_id, movie_id):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor([user_id]).to(device), torch.tensor([movie_id]).to(device))
    return prediction.item()

# 示例预测
sample_user, sample_movie = 1, 2
predicted_rating = predict_rating(sample_user, sample_movie)
print(f'Predicted Rating for user {sample_user} and movie {sample_movie}: {predicted_rating}')
