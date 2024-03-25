import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        # 在自注意力机制中，value、key、query通常是同一个输入
        attention = self.attention(query, key, value)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TransformerRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embed_size, total_rating_classes, heads, forward_expansion, dropout):
        super(TransformerRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.movie_embedding = nn.Embedding(num_movies, embed_size)
        
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        
        self.out = nn.Linear(embed_size, total_rating_classes)  # 假设评分是一个分类问题

    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input)
        # 将用户和电影的嵌入向量合并为序列
        combined = torch.cat((user_embedded.unsqueeze(1), movie_embedded.unsqueeze(1)), dim=1)
        # 应用Transformer块
        x = self.transformer_block(combined, combined, combined)
        # 只取序列的第一个元素，即用户嵌入经过处理后的结果
        out = self.out(x[:, 0, :])
        return out
