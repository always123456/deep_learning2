import numpy as np

layer_grads = [np.zeros((2, 2))]
model_grads = []
model_grads += layer_grads  # 列表的相加，在新列表中存放的元素仍然是原来元素的引用（浅拷贝）

print("更新前, model_grads: ", model_grads)

layer_grads[0][...] = np.array([[1, 2], [3, 4]])

print("更新后, model_grads: ", model_grads)

