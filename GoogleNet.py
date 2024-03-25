class InceptionModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Linear(input_size, output_size // 4)
        self.branch2 = nn.Sequential(
            nn.Linear(input_size, output_size // 4),
            nn.ReLU(),
            nn.Linear(output_size // 4, output_size // 4)
        )
        self.branch3 = nn.Sequential(
            nn.Linear(input_size, output_size // 4),
            nn.ReLU(),
            nn.Linear(output_size // 4, output_size // 4)
        )
        self.branch4 = nn.Linear(input_size, output_size // 4)

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=-1)

class GoogleNetStyleRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(GoogleNetStyleRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        self.inception1 = InceptionModule(embedding_size * 2, 512)
        self.inception2 = InceptionModule(512, 256)
        self.fc_final = nn.Linear(256, 1)

    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input)
        x = torch.cat([user_embedded, movie_embedded], dim=-1)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.fc_final(x)
        return x.squeeze()
