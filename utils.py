import numpy as np
from PIL import Image
from noise import pnoise3
import random
import cv2
from numpy.fft import fft2, ifft2
import skimage.io as io
from skimage import transform
from skimage.color import rgb2gray


def gen_noise(img, depth):
    p1 = Image.new('L', (img.shape[1], img.shape[0]))
    p2 = Image.new('L', (img.shape[1], img.shape[0]))
    p3 = Image.new('L', (img.shape[1], img.shape[0]))

    # scale = 1 / 800.0
    scale = 1 / 130.0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            v = pnoise3(x * scale, y * scale, (depth - random.randint(0, depth)) * scale, octaves=1, persistence=0.5,
                        lacunarity=0.5)
            color = int((v + 1) * 128.0)
            p1.putpixel((x, y), color)

    # scale = 1 / 500.0
    scale = 1 / 60.0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            v = pnoise3(x * scale, y * scale, (depth - random.randint(0, depth)) * scale, octaves=1, persistence=0.5,
                        lacunarity=0.5)
            color = int((v + 0.5) * 128)
            p2.putpixel((x, y), color)

    # scale = 1 / 300.0
    scale = 1 / 10.0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            v = pnoise3(x * scale, y * scale, (depth - random.randint(0, depth)) * scale, octaves=1, persistence=0.5,
                        lacunarity=0.5)
            color = int((v + 1.2) * 128)
            p3.putpixel((x, y), color)

    perlin = (np.array(p1) + np.array(p2) / 2 + np.array(p3) / 4) / 3

    return perlin


def add_stripe(image_path, amount, size, length, beta=0.8):
    image = cv2.imread(image_path)
    angle = random.randint(-30, 30)
    noise = np.random.uniform(0, 256, image.shape[0:2])
    v = amount * 0.01
    noise[np.where(noise < (256 - v))] = 0
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (size, size), 0)
    blurred = cv2.filter2D(noise, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)

    rain = np.expand_dims(blurred, 2)
    # rain_effect = np.concatenate((image, blurred), axis=2)

    rain_result = image.copy()
    rain = np.array(rain, dtype=np.float32)
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]

    cv2.imwrite(image_path, rain_result)


def color_fft(image, flag):
    fre = []
    for i in range(3):
        temp = image[:, :, i]
        if flag:
            fre.append(fft2(temp))
        else:
            fre.append(ifft2(temp))
    fre = np.stack((fre[0], fre[1], fre[2]), axis=2)
    return fre


def overlap(image_path, mask_path):
    image = io.imread(image_path)
    [l, h] = image.shape[0:2]
    fre = color_fft(image, True)
    am = np.abs(fre)
    aa = np.angle(fre)
    int_value = []
    for i in range(3):
        int_value.append(np.trapz(np.trapz(am[:, :, i])))
    mask = io.imread(mask_path)
    mask = rgb2gray(mask)
    mask = transform.resize(mask, (l, h))
    mask_fre = fft2(mask)
    mask_am = np.abs(mask_fre)
    mask_am = np.log(np.abs(mask_am) + 1)
    temp_am = []
    for i in range(3):
        temp_am.append(am[:, :, i] * mask_am)
    temp_am = np.stack((temp_am[0], temp_am[1], temp_am[2]), axis=2)
    for i in range(3):
        int_value_temp = np.trapz(np.trapz(temp_am[:, :, i]))
        temp_am[:, :, i] *= int_value[i] / int_value_temp
    fr1 = temp_am * np.exp(1j * aa)
    image = np.real(color_fft(fr1, False))
    image /= np.max(np.max(image))
    image = (image * 255).astype('uint8')
    io.imsave(image_path, image)


def cal_margin(height, width, point_x, point_y, size):
    lmargin = int(point_x - size / 2)
    umargin = int(point_y - size / 2)
    rmargin = int(width - point_x - size / 2)
    dmargin = int(height - point_y - size / 2)
    return lmargin, umargin, rmargin, dmargin
