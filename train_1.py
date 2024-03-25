#添加预测分数前五的电影ID作为推荐
def recommend_movies(model, user_id, num_users, num_movies, top_k=5):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        # 为指定用户生成所有电影的评分预测
        predictions = model(torch.tensor([user_id] * num_movies).long().to(device), torch.tensor(list(range(num_movies))).long().to(device))
        # 获取评分最高的top_k部电影的索引
        _, top_movie_ids = torch.topk(predictions, k=top_k)
        return top_movie_ids.cpu().numpy()

# 模型训练部分保持不变...

# 在训练结束后使用 recommend_movies 函数为指定用户推荐电影
sample_user_id = 1  # 示例用户ID，可以根据需要更改
recommended_movie_ids = recommend_movies(model, sample_user_id, num_users, num_movies)
print(f"Top 5 recommended movie IDs for user {sample_user_id}: {recommended_movie_ids}")
