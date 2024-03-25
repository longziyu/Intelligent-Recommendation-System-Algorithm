from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
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
    ratings = pd.read_csv('/home/ubuntu/project/recommend/ml-1m/ratings.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
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

num_epochs = 1
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

#添加预测分数前五的电影ID作为推荐
def recommend_movies(model, user_id, num_users, num_movies, top_k=5):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        # 为指定用户生成所有电影的评分预测
        predictions = model(torch.tensor([user_id] * num_movies).long().to(device), torch.tensor(list(range(num_movies))).long().to(device))
        # 获取评分最高的top_k部电影的索引
        _, top_movie_ids = torch.topk(predictions, k=top_k)
        return top_movie_ids.cpu().numpy()

# 在训练结束后使用 recommend_movies 函数为指定用户推荐电影
sample_user_id = 1  # 示例用户ID，可以根据需要更改
recommended_movie_ids = recommend_movies(model, sample_user_id, num_users, num_movies)
print(f"Top 5 recommended movie IDs for user {sample_user_id}: {recommended_movie_ids}")
# 用于收集测试数据和对应的预测结果
actuals = []
predictions = []

# 测试模型并收集数据
model.eval()
with torch.no_grad():
    for batch_idx, (users, movies, ratings) in enumerate(test_loader):
        users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
        preds = model(users, movies)
        
        # 收集真实评分和预测评分
        actuals.extend(ratings.view(-1).cpu().numpy())
        predictions.extend(preds.view(-1).cpu().numpy())

# 计算MSE和MAE
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')

# 将评分转换为二元形式，以便计算精确率、召回率、F1分数和ROC AUC
# 假设评分高于3为正样本（用户喜欢的电影），否则为负样本（用户不喜欢的电影）
binary_actuals = [1 if a > 3.0 else 0 for a in actuals]
binary_predictions = [1 if p > 3.0 else 0 for p in predictions]

precision, recall, f1, _ = precision_recall_fscore_support(binary_actuals, binary_predictions, average='binary')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# 计算ROC曲线和AUC
# 为了计算ROC AUC，我们需要预测评分的概率，而不是具体的评分
# 在这个场景下，我们可以使用模型输出的评分作为正类的概率，但这可能不是最佳实践
fpr, tpr, thresholds = roc_curve(binary_actuals, predictions)
roc_auc = auc(fpr, tpr)

print(f'ROC AUC: {roc_auc:.4f}')
