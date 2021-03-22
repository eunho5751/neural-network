import numpy as np


def im2col(img, kernel_size, strides, padding):
    N, C, H, W = img.shape
    out_h = (H + 2*padding - kernel_size[0]) // strides + 1
    out_w = (W + 2*padding - kernel_size[1]) // strides + 1

    img = np.pad(img, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((N, C, kernel_size[0], kernel_size[1], out_h, out_w))

    for y in range(kernel_size[0]):
        y_max = y + strides * out_h
        for x in range(kernel_size[1]):
            x_max = x + strides * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:strides, x:x_max:strides]

    # N out_h out_w C kernel_size kernel_size -> (N * out_h * out_w, C * kernel_size * kernel_size)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, image_shape, kernel_size, strides=1, padding=0):
    N, C, H, W = image_shape
    out_h = (H + 2 * padding - kernel_size[0]) // strides + 1
    out_w = (W + 2 * padding - kernel_size[1]) // strides + 1
    col = col.reshape(N, out_h, out_w, C, kernel_size[0], kernel_size[1]).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * padding + strides - 1, W + 2 * padding + strides - 1))
    for y in range(kernel_size[0]):
        y_max = y + strides * out_h
        for x in range(kernel_size[1]):
            x_max = x + strides * out_w
            img[:, :, y:y_max:strides, x:x_max:strides] += col[:, :, y, x, :, :]

    return img[:, :, padding:H + padding, padding:W + padding]
