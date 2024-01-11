import torch
import lab1.lr1


def vector_sum_cpu(A):
    count = 0
    lab1.lr1.tic()
    for i in range(len(A)):
        count = count + A[i]
    running_time = lab1.lr1.toc()
    return running_time


def vector_sum_gpu(A):
    tensor = lab1.lr1.mat_copy_gpu(A)
    lab1.lr1.tic()
    count = torch.sum(tensor)
    running_time = lab1.lr1.toc()

    return running_time


if __name__ == '__main__':
    iterations = 200
    step = 5000
    max_size = step * iterations
    min_size = 1000
    SIZE = min_size
    label_x = "Количество элементов"
    label_y = "Время суммирования, сек"

    cpu_timings = []
    gpu_timings = []
    iter_list = []

    while SIZE <= max_size:
        X = lab1.lr1.mat_generator(1, SIZE)
        cpu_timings.append(vector_sum_cpu(X))
        gpu_timings.append(vector_sum_gpu(X))
        iter_list.append(SIZE)
        SIZE = SIZE + step
    lab1.lr1.plotter(cpu_timings, gpu_timings, iter_list, label_x, label_y)
    print("done")