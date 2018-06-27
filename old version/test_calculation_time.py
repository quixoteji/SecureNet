import numpy as np
import time

def vMatrix(A, B, flag):
    '''
    Matrix * Diagnol
    ''' 
    if flag == 1:
        matrix = A
        eye = B
        (rows, cols) = matrix.shape
        res = np.zeros_like(matrix)
        for i in range(rows):
            res[i] = eye[i][i] * matrix[i]
    elif flag == 0:
        matrix = B
        eye = A
        (rows, cols) = matrix.shape
        res = np.zeros_like(matrix)
        for i in range(rows):
            for j in range(cols):
                res[i][j] = eye[i][i] * matrix[i][j]
    elif flag == 2:
        matrix = A 
        eye = B 
        res = np.tile(eye.max(axis=1), (eye.shape[0], 1))
        return res * matrix
    else:
        print('Warning!')
    
    return res

size = 1500
A = np.random.rand(size, size)
B = np.random.rand(size, size)
xA = A * np.eye(size)

time1 = time.time()
xC1 = np.matmul(xA, B) #1
time2 = time.time()
xC2 = np.dot(xA, B)  #2
time3 = time.time()
xC3 = vMatrix(B, xA, 1) # 3
time4 = time.time()
xC4 = vMatrix(xA, B, 0) # 4
time5 = time.time()
XC5 = vMatrix(B, xA, 2) # 5
time6 = time.time()

print(time2 - time1) #1
print(time3 - time2) #2
print(time4 - time3) #3
print(time5 - time4) #4
print(time6 - time5) #5


# 4 > 1 = 2 > 5 > 3
# 4 > 2 > 1 > 5 > 3


