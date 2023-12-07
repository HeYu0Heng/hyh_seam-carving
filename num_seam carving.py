import numpy as np
demo = np.random.randint(1,10,(5,5))
print('输出的初始随机矩阵', '\n', demo)

def energy(matrix):
    num_row, num_col = matrix.shape
    a = np.zeros_like(matrix)
    a[0, :] = matrix[0, :]
    row = 1
    while row < num_row:
        for j in range(num_col):
            left = matrix[row - 1, j - 1] if j > 0 else float('inf')
            middle = matrix[row - 1, j]
            right = matrix[row - 1, j + 1] if j < num_col - 1 else float('inf')
            a[row, j] = matrix[row, j] + min(left, middle, right)
            matrix[row,j] += min(left, middle, right)
        row += 1
    return a
outcome = energy(demo)
print('输出的全局最优能量矩阵', '\n', outcome)




outcome_row, outcome_col = outcome.shape
path_sum = np.min(outcome[outcome_col-1, :])
print('最短路径和',path_sum)
for i in range(outcome_row-1,outcome_row):
    for j in range(outcome_col):
        if path_sum == outcome[i, j]:
            start_col = j
            print('回溯起始列', start_col)
            break



def backtracking(matrix, row, col):
    num_row, num_col = matrix.shape
    path_point = [(row, col)]
    while row > 0:
        up = matrix[row - 1, col] if row - 1 >= 0 else float('inf')
        left_up = matrix[row - 1, col - 1] if row - 1 >= 0 and col - 1 >= 0 else float('inf')
        right_up = matrix[row - 1, col + 1] if row - 1 >= 0 and col + 1 < num_col else float('inf')
        min_num = min(up, right_up, left_up)
        if up == min_num:
            row -= 1
        elif left_up == min_num:
            row -= 1
            col -= 1
        else:
            row -= 1
            col += 1
        path_point.append((row,col))
    return path_point

path_way =  backtracking(outcome,outcome_row-1,start_col)
print('最优路径', path_way)


