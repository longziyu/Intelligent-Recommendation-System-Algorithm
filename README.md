这是一个关于推荐系统算法的毕业设计项目（不是我的）

/usr/bin/nvidia-modprobe: unrecognized option: "-s"

ERROR: Invalid commandline, please run `/usr/bin/nvidia-modprobe --help` for usage information.

/usr/bin/nvidia-modprobe: unrecognized option: "-s"

ERROR: Invalid commandline, please run `/usr/bin/nvidia-modprobe --help` for usage information.

Traceback (most recent call last):
  File "/home/violeteyes/miniconda3/envs/env/lib/python3.7/site-packages/pandas/core/indexes/base.py", line 3361, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 76, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 108, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 2131, in pandas._libs.hashtable.Int64HashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 2140, in pandas._libs.hashtable.Int64HashTable.get_item
KeyError: 459748

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/media/violeteyes/wings/project/recommend/train.py", line 65, in <module>
    for users, movies, ratings in train_loader:
  File "/home/violeteyes/miniconda3/envs/env/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 628, in __next__
    data = self._next_data()
  File "/home/violeteyes/miniconda3/envs/env/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 671, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/violeteyes/miniconda3/envs/env/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 58, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/violeteyes/miniconda3/envs/env/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 58, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/media/violeteyes/wings/project/recommend/train.py", line 17, in __getitem__
    return self.users[idx], self.movies[idx], self.ratings[idx]
  File "/home/violeteyes/miniconda3/envs/env/lib/python3.7/site-packages/pandas/core/series.py", line 942, in __getitem__
    return self._get_value(key)
  File "/home/violeteyes/miniconda3/envs/env/lib/python3.7/site-packages/pandas/core/series.py", line 1051, in _get_value
    loc = self.index.get_loc(label)
  File "/home/violeteyes/miniconda3/envs/env/lib/python3.7/site-packages/pandas/core/indexes/base.py", line 3363, in get_loc
    raise KeyError(key) from err
KeyError: 459748
