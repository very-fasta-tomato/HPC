import numpy as np
import numba
import cv2
import lab1.lr1

@numba.njit
def bilinear_interpolation_gpu(image, output_shape, timer_on=True):
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = output_shape[:2]

    # Compute scale factors
    sh = src_h / dst_h
    sw = src_w / dst_w

    # Create output image
    output_image = np.zeros(output_shape, dtype=np.float32)

    for i in range(dst_h):
        for j in range(dst_w):
            # Calculate the coordinate in the source image
            y = i * sh
            x = j * sw

            # Compute the integer part of the coordinates
            y_int = int(y)
            x_int = int(x)

            # Compute the fractional part of the coordinates
            y_frac = y - y_int
            x_frac = x - x_int

            y_int_next = min(y_int + 1, src_h - 1)
            x_int_next = min(x_int + 1, src_w - 1)

            # Perform bilinear interpolation
            output_image[i, j] = (
                (1 - x_frac) * (1 - y_frac) * image[y_int, x_int] +
                (1 - x_frac) * y_frac * image[y_int_next, x_int] +
                x_frac * (1 - y_frac) * image[y_int, x_int_next] +
                x_frac * y_frac * image[y_int_next, x_int_next]
            )

    return output_image


def bilinear_interpolation_cpu(image, new_size, timer_on=True):
    # Получаем размеры исходного и нового изображений
    height, width = image.shape[:2]
    new_height, new_width = new_size

    # Вычисляем коэффициенты масштабирования по каждой оси
    scale_x = float(width) / new_width
    scale_y = float(height) / new_height

    # Инициализируем новое изображение заданного размера
    interpolation_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    lab1.lr1.tic()

    for y in range(new_height):
        for x in range(new_width):
            # Находим координаты соответствующие текущей позиции в новом изображении
            src_x = (x + 0.5) * scale_x - 0.5
            src_y = (y + 0.5) * scale_y - 0.5

            # Находим координаты ближайших пикселей в исходном изображении
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, width - 1)
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, height - 1)

            # Находим весовые коэффициенты для билинейной интерполяции
            dx = src_x - x0
            dy = src_y - y0

            # Вычисляем значения пикселей в новом изображении с помощью билинейной интерполяции
            interpolation_image[y, x] = (1 - dx) * (1 - dy) * image[y0, x0] \
                                        + dx * (1 - dy) * image[y0, x1] \
                                        + (1 - dx) * dy * image[y1, x0] \
                                        + dx * dy * image[y1, x1]

    running_time = lab1.lr1.toc()
    if timer_on == True:
        return running_time
    else:
        return interpolation_image


def time_gpu(resized_image, new_size):
    lab1.lr1.tic()
    bilinear_interpolation_gpu(resized_image, new_size)
    time = lab1.lr1.toc()
    return time


def time_cpu(image, new_size):
    lab1.lr1.tic()
    bilinear_interpolation_cpu(image, new_size, False)
    time = lab1.lr1.toc()
    return time

'''
# Загрузка изображения
image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

# Указание нового размера (увеличение в 2 раза)
new_size = (image.shape[0] * 2, image.shape[1] * 2)
print(new_size)

# Изменение размера изображения
resized_image = cv2.resize(image, new_size)

# Выполнение билинейной интерполяции
output_image = bilinear_interpolation(resized_image, new_size)

# Сохранение исходного и интерполированного изображений
cv2.imwrite('InterpolatedImageCPU.jpg', output_image)
cv2.imwrite('InterpolatedImageGPU.jpg', output_image)
'''

if __name__ == '__main__':
    iterations = 20
    size_mult = 2
    label_x = "Коэфициэнт интерполяции"
    label_y = "Время суммирования элементов, сек"

    cpu_timings = []
    gpu_timings = []
    iter_list = []
    # Загрузка изображения
    image = cv2.imread('test.png')
    image2 = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    # Указание нового размера (увеличение в 2 раза)
    new_size = (image.shape[0] * size_mult, image.shape[1] * size_mult)
    print(new_size)

    # Изменение размера изображения
    resized_image = cv2.resize(image, new_size)

    for i in range(iterations):
        # Указание нового размера (увеличение в 2 раза)
        new_size = (image.shape[0] * size_mult, image.shape[1] * size_mult)
        print(new_size)
        resized_image = cv2.resize(image2, new_size)
        cpu_timings.append(time_cpu(image, new_size))
        gpu_timings.append(time_gpu(resized_image, new_size))
        iter_list.append(i)
        size_mult = size_mult + 1
    lab1.lr1.plotter(cpu_timings, gpu_timings, iter_list, label_x, label_y)
    print("done")