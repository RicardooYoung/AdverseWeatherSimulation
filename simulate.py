import os
import time
import argparse
import numpy as np

import render


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--fog', type=bool, default=False)
    parser.add_argument('--rain', type=bool, default=False)
    parser.add_argument('--smoke', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    path = args.path
    fog = args.fog
    rain = args.rain
    smoke = args.smoke
    raw_path = os.path.join(path, 'Raw')
    fog_path = os.path.join(path, 'Fog')
    if not os.path.exists(fog_path):
        os.mkdir(fog_path)
    rain_path = os.path.join(path, 'Rain')
    if not os.path.exists(rain_path):
        os.mkdir(rain_path)
    smoke_path = os.path.join(path, 'Smoke')
    if not os.path.exists(smoke_path):
        os.mkdir(smoke_path)
    label_path = os.path.join(path, 'Label')
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    pattern_path = 'SmokePattern'
    image_list = os.listdir(raw_path)

    print('Detect {} files.'.format(len(image_list)))

    my_render = render.Render()
    my_render.set_raw_path(raw_path)
    my_render.set_fog_path(fog_path)
    my_render.set_rain_path(rain_path)
    my_render.set_smoke_path(smoke_path)
    my_render.set_pattern_path(pattern_path)
    my_render.set_label_path(label_path)


    # pattern_list = os.listdir(pattern_path)
    #
    # index = np.random.randint(0, len(pattern_list), dtype=int)

    color_list = [240, 30]
    color_flag = ['white', 'black']

    for i in range(len(image_list)):
        if image_list[i].startswith('.'):
            print('Detect a system generated file, pass.')
            continue
        print('[{}/{}]Now processing {}.'.format(i + 1, len(image_list), image_list[i].split('\\')[-1]))
        start_time = time.time()
        image_path = os.path.join(raw_path, image_list[i])
        my_render.read_image(image_path)
        if fog:
            print('|_Fog simulation started.')
            my_render.add_fog()
        if rain:
            print('|_Rain simulation started.')
            my_render.add_rain()
        if smoke:
            print('|_Smoke simulation started.')
            my_render.label_parse()
            # chosen_pattern_path = os.path.join(pattern_path, pattern_list[index])
            # print(' |_Choose smoke pattern <{}>'.format(pattern_list[index]))
            for j in range(2):
                my_render.set_smoke_color(color_list[j], color_flag[j])
                my_render.add_smoke()
        end_time = time.time()
        print('|_Time elapsed {:.3f}s'.format(end_time - start_time))

    print('All finished!')
