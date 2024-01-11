import numpy as np
import torch
import matplotlib.pyplot as plt


def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        return time.time() - startTime_for_tictoc
    else:
        print ("Toc: start time not set")


def multiply_matrix(A, B):
    shape = int(len(A)), int(len(B[0]))
    if len(A) < len(B):
        exception = "Размерность не совпадает"
        return exception
    result = np.zeros(shape)
    tic()
    for i in range(len(A)):
        # iterate through columns of B
        for j in range(len(B[0])):
            # iterate through rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    running_time = toc()
    return running_time


def multiply_matrix_gpu(A, B):
    if len(A) < len(B):
        exception = "Размерность не совпадает"
        return exception
    tic()
    result = torch.matmul(A, B)
    print(result)
    running_time = toc()
    return running_time


def mat_generator(size_x, size_y, int_range=10):
    array = np.random.randint(int_range, size=(size_y, size_x))
    # print(array)
    return array


def mat_copy_gpu(matrix):
    tensor1 = torch.from_numpy(matrix)
    return tensor1


def plotter(cpu_timings, gpu_timings, iter_list, label_x, label_y):
    plt.plot(iter_list, cpu_timings, label='CPU')
    plt.legend()
    plt.plot(iter_list, gpu_timings, label='GPU')
    plt.legend()
    plt.grid()
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    plt.savefig("LR1figure.png")
    plt.show()


if __name__ == '__main__':

    max_size = 1000
    min_size = 10
    SIZE = min_size
    step = 100

    cpu_timings = []
    gpu_timings = []
    iter_list = []

    label_x = "Размерность матриц"
    label_y = "Время, сек"

    while SIZE <= max_size:
        X = mat_generator(SIZE, SIZE)
        Y = mat_generator(SIZE, SIZE)
        XG = mat_copy_gpu(X)
        YG = mat_copy_gpu(Y)
        cpu_timings.append(multiply_matrix(X, Y))
        gpu_timings.append(multiply_matrix_gpu(XG, YG))
        iter_list.append(SIZE)
        SIZE = SIZE + step
    plotter(cpu_timings, gpu_timings, iter_list, label_x, label_y)