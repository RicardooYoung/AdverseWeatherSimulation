import os
import cv2
import numpy as np
from utils import gen_noise, add_stripe, overlap, cal_margin

import skimage.io as io
from skimage.color import rgb2gray
from skimage import transform

np.set_printoptions(threshold=np.inf)
np.errstate(invalid='ignore', divide='ignore')


class Render:
    def __init__(self):
        self.cam_height = 1000
        self.fog_height = 70
        self.haze_height = 35
        self.fog_visibility = 100
        self.haze_visibility = 450
        # define hyper-parameter
        self.condition_dict = {0: 'heavy', 1: 'medium', 2: 'light'}
        self.fog_visibility_sequence = [100, 200, 350]
        self.smoke_visibility_sequence = [100, 200, 400]
        self.coverage_ratio = [10, 25, 40]

        self.image = None
        self.image_name = None
        self.raw_path = None
        self.label_path = None
        # input path
        self.fog_path = None
        self.rain_path = None
        self.smoke_path = None
        self.cloud_path = None
        self.pattern_path = None
        self.result_path = None
        self.mask_path = None
        # output path
        self.chosen_label_path = None
        self.height = 0
        self.width = 0
        self.mask = None
        self.label = []
        self.input_channels = 3

        self.render_type = None

        self.fog_color = []
        self.rain_color = []
        self.smoke_color = []
        self.color_set = False
        self.color_flag = ''
        self.point = []
        self.size = []
        self.direction = []

    def set_raw_path(self, raw_path):
        self.raw_path = raw_path

    def set_fog_path(self, fog_path):
        self.fog_path = fog_path

    def set_rain_path(self, rain_path):
        self.rain_path = rain_path

    def set_smoke_path(self, smoke_path):
        self.smoke_path = smoke_path

    def set_cloud_path(self, cloud_path):
        self.cloud_path = cloud_path

    def set_pattern_path(self, pattern_path):
        self.pattern_path = pattern_path

    def set_label_path(self, label_path):
        self.label_path = label_path

    def set_mask_path(self, mask_path):
        self.mask_path = mask_path

    def read_image(self, image_path):
        self.image = cv2.imread(image_path)
        self.image_name = image_path.split('\\')[-1].split('.')[0]
        self.height, self.width = self.image.shape[:2]
        if len(self.image.shape) == 3:
            self.input_channels = 3
        elif len(self.image.shape) == 2:
            self.input_channels = 1
        scale = [256, 512, 1024, 2048, 4096]
        margin = np.zeros_like(scale)
        for i in range(len(scale)):
            margin[i] = np.abs(scale[i] - np.max((self.height, self.width)))
        index = np.argmin(margin)
        self.mask = os.path.join(self.mask_path, 'mask_{}.jpg'.format(scale[index]))
        self.fog_color = np.ones_like(self.image)
        self.fog_color *= 240
        self.rain_color = np.ones_like(self.image)
        self.rain_color[:, :, 0] = 225
        self.rain_color[:, :, 1] = 225
        self.rain_color[:, :, 2] = 201
        self.smoke_color = np.ones_like(self.image)

    def set_smoke_color(self, smoke_color, color_flag):
        self.smoke_color = np.ones_like(self.image) * smoke_color
        self.color_flag = color_flag
        self.color_set = True

    def label_parse(self):
        self.chosen_label_path = os.path.join(self.label_path, self.image_name)
        self.chosen_label_path += '.txt'
        with open(self.chosen_label_path, 'r') as f:
            lines = f.readlines()
        self.label = []
        self.point = []
        self.size = []
        self.direction = []
        for line in lines:
            temp = line[:-1].split(' ')
            temp_x = float(temp[2])
            temp_y = float(temp[3])
            if temp_x < 0:
                temp_x = 0.01
            elif temp_x > 1:
                temp_x = 0.99
            if temp_y < 0:
                temp_y = 0.01
            elif temp_y > 1:
                temp_y = 0.99
            point_x = int(self.width * temp_x)
            point_y = int(self.height * temp_y)
            self.point.append([point_x, point_y])
            length_x = float(temp[4])
            length_y = float(temp[5])
            self.size.append(int((self.width * length_x + self.height * length_y) / 2.5))
            self.label.append([length_x, length_y])
            if float(temp[3]) >= float(temp[4]):
                self.direction.append('horizontal')
            else:
                self.direction.append('vertical')

    def render_mask(self, **kwargs):
        temp_ecf = 3.912 / self.fog_visibility
        temp_ech = 3.912 / self.haze_visibility

        shader = np.empty_like(self.image)

        elevation = np.ones((self.height, self.width))
        elevation *= self.cam_height

        if self.fog_height != 0:
            if self.render_type == 'fog':
                perlin = gen_noise(self.image, self.cam_height)
            ech = temp_ech
            c = (1 - elevation / (self.fog_height + 0.00001))
            c[c < 0] = 0

            if self.fog_height > self.haze_height:
                if self.render_type == 'fog':
                    ecf = (temp_ecf * c + (1 - c) * temp_ech) * (perlin / 255)
                else:
                    ecf = temp_ecf * c + (1 - c) * temp_ech
            else:
                if self.render_type == 'fog':
                    ecf = (temp_ech * c + (1 - c) * temp_ecf) * (perlin / 255)
                else:
                    ecf = temp_ech * c + (1 - c) * temp_ecf

        else:
            ech = temp_ech
            ecf = temp_ecf

        if 'depth_map' in kwargs:
            depth_map = kwargs['depth_map']
            distance_through_fog = np.zeros_like(elevation) + depth_map
            distance_through_haze = np.zeros_like(elevation) + depth_map
            distance_through_fog *= self.fog_height
            distance_through_haze *= self.haze_height
        else:
            distance_through_fog = np.ones_like(elevation)
            distance_through_haze = np.ones_like(elevation)
            distance_through_fog *= self.fog_height
            distance_through_haze *= self.haze_height

        if self.input_channels == 3:
            shader[:, :, 0] = self.image[:, :, 0] * np.exp(-ech * distance_through_haze - ecf * distance_through_fog)
            shader[:, :, 1] = self.image[:, :, 1] * np.exp(-ech * distance_through_haze - ecf * distance_through_fog)
            shader[:, :, 2] = self.image[:, :, 2] * np.exp(-ech * distance_through_haze - ecf * distance_through_fog)
        else:
            shader = self.image * np.exp(-ech * distance_through_haze - ecf * distance_through_fog)
        omit = 1 - np.exp(-ech * distance_through_haze - ecf * distance_through_fog)

        return shader, omit

    def synthesizer(self, **kwargs):
        if self.render_type != 'smoke' and self.render_type != 'cloud':
            if self.render_type == 'fog':
                index = kwargs['index']
                self.result_path = os.path.join(self.fog_path, self.image_name)
                self.result_path += '_{}_fog.png'.format(self.condition_dict[index])
                color = self.fog_color
            elif self.render_type == 'rain':
                self.result_path = os.path.join(self.rain_path, self.image_name)
                self.result_path += '_rain.png'
                color = self.rain_color
            shader, omit = self.render_mask()
        else:
            if self.render_type == 'smoke':
                depth_map = kwargs['depth_map']
                index = kwargs['index']
                self.result_path = os.path.join(self.smoke_path, self.image_name)
                self.result_path += '_{}_{}_smoke.png'.format(self.condition_dict[index], self.color_flag)
                color = self.smoke_color
                shader, omit = self.render_mask(depth_map=depth_map)
            else:
                depth_map = kwargs['depth_map']
                self.result_path = os.path.join(self.cloud_path, self.image_name)
                self.result_path += '_cloud.png'
                color = self.smoke_color
                shader, omit = self.render_mask(depth_map=depth_map)

        result = np.empty_like(self.image)

        result[:, :, 0] = shader[:, :, 0] + omit * color[:, :, 0]
        result[:, :, 1] = shader[:, :, 1] + omit * color[:, :, 1]
        result[:, :, 2] = shader[:, :, 2] + omit * color[:, :, 2]
        cv2.imwrite(self.result_path, result)

    def gen_pattern(self, point_x, point_y, size, ratio, pattern_path, label, direction):
        if label[0] > 0.9 or label[1] > 0.9:
            return np.zeros((self.height, self.width))
        if size % 2 != 0:
            size -= 1
        if point_x <= int(size / 2) + 1:
            point_x += int(size / 2) + 1
        elif point_x + int(size / 2) + 1 >= self.width:
            point_x -= int(size / 2) + 1
        if point_y <= int(size / 2) + 1:
            point_y += int(size / 2) + 1
        elif point_y + int(size / 2) + 1 >= self.height:
            point_y -= int(size / 2) + 1
        pattern = io.imread(pattern_path)
        if len(pattern.shape) == 3:
            pattern = rgb2gray(pattern)
        pattern = transform.resize(pattern, (size, size))
        lmargin, umargin, rmargin, dmargin = cal_margin(self.height, self.width, point_x, point_y, size)
        if direction == 'horizontal':
            bias = int((100 - ratio) / 100 * self.width * label[0] / 2)
            if lmargin >= rmargin:
                if point_x - bias > int(size / 2) + 1:
                    point_x -= bias
                else:
                    point_x -= point_x - int(size / 2) - 1
            else:
                if point_x + bias < self.width - int(size / 2) - 1:
                    point_x += bias
                else:
                    point_x += self.width - point_x - int(size / 2) - 1
        else:
            bias = int((100 - ratio) / 100 * self.height * label[1] / 2)
            if umargin >= dmargin:
                if point_y - bias > int(size / 2) + 1:
                    point_y -= bias
                else:
                    point_y -= point_y - int(size / 2) - 1
            else:
                if point_y + bias < self.height - int(size / 2) - 1:
                    point_y += bias
                else:
                    point_y += self.height - point_y + int(size / 2) - 1
        lmargin, umargin, rmargin, dmargin = cal_margin(self.height, self.width, point_x, point_y, size)
        depth_map = np.pad(pattern, ((umargin, dmargin), (lmargin, rmargin)), 'constant')
        return depth_map

    def gen_point(self):
        num_cloud = np.random.randint(3, 6)
        self.point = []
        self.size = []
        self.direction = []
        temp_x_list = (np.random.rand(num_cloud) - 0.5)*np.pi
        temp_x_list = (np.sin(temp_x_list) + 1) / 2
        temp_x_list = (temp_x_list - 0.5) * np.pi
        temp_x_list = (np.sin(temp_x_list) + 1) / 2
        temp_y_list = (np.random.rand(num_cloud) - 0.5)*np.pi
        temp_y_list = (np.sin(temp_y_list) + 1) / 2
        temp_y_list = (temp_y_list - 0.5) * np.pi
        temp_y_list = (np.sin(temp_y_list) + 1) / 2
        length_x_list = np.random.rand(num_cloud) / 2 + 0.3
        length_y_list = np.random.rand(num_cloud) / 2 + 0.3
        for i in range(num_cloud):
            temp_x = temp_x_list[i]
            temp_y = temp_y_list[i]
            length_x = length_x_list[i]
            length_y = length_y_list[i]
            if temp_x < 0:
                temp_x = 0.01
            elif temp_x > 1:
                temp_x = 0.99
            if temp_y < 0:
                temp_y = 0.01
            elif temp_y > 1:
                temp_y = 0.99
            point_x = int(self.width * temp_x)
            point_y = int(self.height * temp_y)
            self.point.append([point_x, point_y])
            self.size.append(int((self.width * length_x + self.height * length_y) / 2.5))
            self.label.append([length_x, length_y])
            if length_x > length_y:
                self.direction.append('horizontal')
            else:
                self.direction.append('vertical')

    def add_fog(self, heavy, medium, light):
        self.render_type = 'fog'
        choice = [heavy, medium, light]
        for index in range(len(self.fog_visibility_sequence)):
            if choice[index]:
                self.haze_visibility = self.fog_visibility_sequence[index]
                self.synthesizer(index=index)

    def add_rain(self):
        self.render_type = 'rain'
        self.haze_visibility = 450
        self.synthesizer()
        overlap(self.result_path, self.mask)
        add_stripe(self.result_path, 60, 5, 50)

    def add_smoke(self, heavy, medium, light):
        self.render_type = 'smoke'
        depth_map = np.zeros((self.height, self.width))
        choice = [heavy, medium, light]
        for i in range(len(self.point)):
            if self.size[i] / self.width >= 0.75 or self.size[i] / self.height >= 0.75:
                continue
            pattern_list = os.listdir(self.pattern_path)
            index = np.random.randint(0, len(pattern_list), dtype=int)
            chosen_pattern_path = os.path.join(self.pattern_path, pattern_list[index])
            depth_map += self.gen_pattern(self.point[i][0], self.point[i][1], self.size[i], 80,
                                          chosen_pattern_path, self.label[i], self.direction[i])
            for index in range(len(self.smoke_visibility_sequence)):
                if choice[index]:
                    self.haze_visibility = self.smoke_visibility_sequence[index]
                    self.synthesizer(depth_map=depth_map, index=index)

    def add_cloud(self):
        self.render_type = 'cloud'
        depth_map = np.zeros((self.height, self.width))
        self.gen_point()
        for i in range(len(self.point)):
            pattern_list = os.listdir(self.pattern_path)
            index = np.random.randint(0, len(pattern_list), dtype=int)
            chosen_pattern_path = os.path.join(self.pattern_path, pattern_list[index])
            depth_map += self.gen_pattern(self.point[i][0], self.point[i][1], self.size[i], 80,
                                          chosen_pattern_path, self.label[i], self.direction[i])
            self.haze_visibility = self.smoke_visibility_sequence[1]
            self.synthesizer(depth_map=depth_map)
