这是一个关于推荐系统算法的符合本科毕业设计难度的项目（不是我的毕设）

直接运行train.py即可（注意修改data的文件路径）

我使用了 cmap='viridis'。你也可以尝试其他色图，例如 cmap='plasma'、cmap='inferno' 或 cmap='magma'，看看哪种色图最适合你的数据。

训练结束后在终端运行：tensorboard --logdir=runs查看训练情况

实验结果：

![image](https://github.com/longziyu/Intelligent-Recommendation-System-Algorithm/blob/main/predict.png)

loss曲线如下：

![image](https://github.com/longziyu/Intelligent-Recommendation-System-Algorithm/blob/main/loss.png)
![image](https://github.com/longziyu/Intelligent-Recommendation-System-Algorithm/blob/main/training_loss.png)


  File "/home/ubuntu/miniconda3/envs/carla/lib/python3.7/site-packages/torch/nn/functional.py", line 2210, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.ShortTensor instead (while checking arguments for embedding)
