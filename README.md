这是一个关于推荐系统算法的毕业设计项目（不是我的）

Traceback (most recent call last):
  File "/media/violeteyes/wings/project/recommend/train.py", line 73, in <module>
    outputs = model(users, movies)
  File "/home/violeteyes/miniconda3/envs/carla/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/media/violeteyes/wings/project/recommend/train.py", line 41, in forward
    user_embedded = self.user_embedding(user_input)
  File "/home/violeteyes/miniconda3/envs/carla/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/violeteyes/miniconda3/envs/carla/lib/python3.7/site-packages/torch/nn/modules/sparse.py", line 160, in forward
    self.norm_type, self.scale_grad_by_freq, self.sparse)
  File "/home/violeteyes/miniconda3/envs/carla/lib/python3.7/site-packages/torch/nn/functional.py", line 2199, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.ShortTensor instead (while checking arguments for embedding)
