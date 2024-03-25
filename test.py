from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

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
