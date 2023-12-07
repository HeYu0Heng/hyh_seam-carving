import numpy as np
import cv2

img = cv2.imread('demo1.jpg',1)
cv2.imshow('picture',img)
cv2.waitKey()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_X = cv2.Sobel(img_gray,cv2.CV_64F,1,0)
img_gray_Y = cv2.Sobel(img_gray,cv2.CV_64F,0,1)
img_gray_X = cv2.convertScaleAbs(img_gray_X)
img_gray_Y = cv2.convertScaleAbs(img_gray_Y)
result_image = cv2.addWeighted(img_gray_X,0.5,img_gray_Y,0.5,0)

result_image = result_image.astype(np.uint64)
def energy(matrix):
    num_row, num_col = matrix.shape
    a = np.zeros_like(matrix)
    a[0, :] = matrix[0, :]
    row = 1
    while row < num_row:
        for j in range(num_col):
            left = matrix[row - 1, j - 1] if j > 0 else np.inf
            middle = matrix[row - 1, j]
            right = matrix[row - 1, j + 1] if j < num_col - 1 else np.inf
            a[row, j] = matrix[row, j] + min(left, middle, right)
            matrix[row, j] += min(left, middle, right)
        row += 1
    return a

outcome = energy(result_image)
print(outcome)


outcome_row, outcome_col = outcome.shape
print(outcome.shape)
path_sum = np.min(outcome[outcome_row-1, :])
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

path_way = backtracking(outcome, outcome_row-1, start_col)
print('最优路径', path_way)
def draw_path(image, path_points):
    for point in path_points:
        x, y = point
        cv2.circle(image, (y, x),int(0.5) ,(0, 0, 255), -1)
    return image


finally_img = draw_path(img,path_way)
print(finally_img.shape)
cv2.imshow('outcome', finally_img)
cv2.waitKey(0)


def crop(img, path):
    r, c, channel = img.shape
    mask = np.ones((r, c), dtype=np.bool_)
    for i, j in path:
        mask[i, j] = False

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img


cropped_image = crop(img, path_way)
print(cropped_image.shape)
cv2.imshow('cropped_image', cropped_image)
cv2.waitKey(0)

