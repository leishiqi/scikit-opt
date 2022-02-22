'''
min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
s.t.
    x1*x2 >= 1
    x1*x2 <= 5
    x2 + x3 = 1
    0 <= x1, x2, x3 <= 5
'''

import numpy as np
from sko.DE import DE
import math


def obj_func(p):
    dim = p.shape[1]
    pop = p.shape[0]
    sorted_array = np.argsort(p, axis=1)
    matrix = np.ones([dim, 30])
    local_temp = [26.8, 25, 24.6, 26.4, 24.3, 23.4, 25, 24.4, 21.3, 24.6, 24.5, 24.1, 24.6, 24.5, 24.1, 25, 24.4, 21.3,
                  26.4, 24.3, 23.4, 26.8, 25, 24.6, 23.4, 21.5, 21.8, 22.3, 23.8, 24.2]
    profile = np.ones([dim, 4])
    # 获取舒适概率矩阵
    for i in range(dim):
        for j in range(30):
            matrix[i][j] = 1 / (1 + math.exp(profile[i][0] + profile[i][1] * local_temp[j]) + math.exp(
                profile[i][2] + profile[i][3] * local_temp[j]))
    prob = np.ones(pop)
    # 解码，计算每个population的平均热舒适概率值
    for i in range(pop):
        cur_prob = 0
        copy_matrix = matrix.copy()
        for j in range(dim):
            index = np.argmax(copy_matrix[sorted_array[i][j]])
            cur_prob += copy_matrix[sorted_array[i][j]][index]
            copy_matrix[:, index] = np.zeros(dim)
        prob[i] = cur_prob / dim
    return prob


# %% Do DifferentialEvolution
de = DE(func=obj_func, n_dim=15, size_pop=50, max_iter=800, lb=0, ub=1)
best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
