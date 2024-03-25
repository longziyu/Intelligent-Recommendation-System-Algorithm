import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
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
    ratings['UserID'] = ratings['UserID'].astype("category").cat.codes
    ratings['MovieID'] = ratings['MovieID'].astype("category").cat.codes
    return ratings

# 加载电影数据并创建电影ID到名称的映射
def load_movie_titles(path_to_movies_dat):
    movies = pd.read_csv(path_to_movies_dat, sep='::', header=None, names=['MovieID', 'Title', 'Genres'], engine='python')
    # 注意：这里不对MovieID重新编码，因为我们将使用原始ID来构建映射
    movie_id_to_name = pd.Series(movies.Title.values, index=movies.MovieID).to_dict()
    return movie_id_to_name

# 模型定义
class RecommenderNet(nn.Module):
    # ... 保持原有的模型定义不变 ...

# 数据加载和处理
ratings = load_data()
movie_id_to_name = load_movie_titles('path_to_movielens_1m/movies.dat')
# ... 保持数据加载和处理的代码不变 ...

# 预测评分矩阵
def predict_matrix(model, num_users, num_movies):
    # ... 保持预测函数的代码不变 ...

# 生成预测矩阵
pred_matrix = predict_matrix(model, 50, 50)

# 绘制热图，使用电影名称作为X轴标签
plt.figure(figsize=(20, 15))  # 可能需要调整大小以适应所有电影名称
plt.imshow(pred_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('User-Movie Rating Predictions')
movie_names = [movie_id_to_name.get(i) for i in range(num_movies)]  # 获取电影名称
plt.xticks(ticks=np.arange(num_movies), labels=movie_names, rotation=90)
plt.xlabel('Movie Name')
plt.ylabel('User ID')
plt.tight_layout()
plt.show()
