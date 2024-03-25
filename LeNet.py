class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(LeNetStyleRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        # 将嵌入层的输出视为1x1的"图像"，其中通道数等于两个嵌入的总长度
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(1, embedding_size*2), stride=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 1), stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(1, 1), stride=1)
        
        # 计算conv2的输出大小以匹配全连接层
        self.fc1 = nn.Linear(16 * 1 * 1, 120)  # 注意：这里的尺寸需要根据实际情况调整
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input)
        x = torch.cat([user_embedded, movie_embedded], dim=-1)  # Concatenate embeddings
        x = x.view(-1, 1, 1, embedding_size * 2)  # Reshape to (batch_size, channels, height, width)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 1 * 1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()
