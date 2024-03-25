class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, use_dropout=False):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(output_size, output_size)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        
        # 如果输入和输出大小不同，需要一个转换层来匹配维度
        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        
        out += identity  # 将输入添加到残差块的输出上
        out = self.relu(out)  # 再次应用激活函数
        return out

class ResNetStyleRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(ResNetStyleRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        # 定义一个残差块序列
        self.res_blocks = nn.Sequential(
            ResidualBlock(embedding_size * 2, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128, use_dropout=True)  # 例子中添加一个带Dropout的残差块
        )
        
        self.fc_final = nn.Linear(128, 1)  # 最终的输出层

    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input)
        x = torch.cat([user_embedded, movie_embedded], dim=-1)
        
        x = self.res_blocks(x)
        x = self.fc_final(x)
        return x.squeeze()
