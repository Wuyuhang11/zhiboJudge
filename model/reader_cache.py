import os
import pickle

# 1.保存到缓存中
def save_to_cache(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# 2.从缓存中加载
def load_from_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None