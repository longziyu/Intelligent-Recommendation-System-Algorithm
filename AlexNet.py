class AlexNetStyleRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(AlexNetStyleRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        # 基于AlexNet的全连接层架构
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_size * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input)
        x = torch.cat([user_embedded, movie_embedded], dim=-1)
        x = self.fc_layers(x)
        return x.squeeze()
