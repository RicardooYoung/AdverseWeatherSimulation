import os
import time
import argparse

import render


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foggy', default=False, action='store_true')
    parser.add_argument('--rainy', default=False, action='store_true')
    parser.add_argument('--smoky', default=False, action='store_true')
    parser.add_argument('--light', default=False, action='store_true')
    parser.add_argument('--medium', default=False, action='store_true')
    parser.add_argument('--heavy', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    path = 'data'
    foggy = args.foggy
    rainy = args.rainy
    smoky = args.smoky

    raw_path = os.path.join(path, 'raw')
    mask_path = os.path.join(path, 'masks')
    pattern_path = os.path.join(path, 'patterns')

    fog_path = os.path.join(path, 'foggy')
    if not os.path.exists(fog_path):
        os.mkdir(fog_path)
    rain_path = os.path.join(path, 'rainy')
    if not os.path.exists(rain_path):
        os.mkdir(rain_path)
    smoke_path = os.path.join(path, 'smoky')
    if not os.path.exists(smoke_path):
        os.mkdir(smoke_path)

    label_path = os.path.join(path, 'labels')
    pattern_path = os.path.join(path, 'patterns')

    image_list = os.listdir(raw_path)

    print('Detect {} files.'.format(len(image_list)))

    my_render = render.Render()
    my_render.set_raw_path(raw_path)
    my_render.set_fog_path(fog_path)
    my_render.set_rain_path(rain_path)
    my_render.set_smoke_path(smoke_path)
    my_render.set_mask_path(mask_path)
    my_render.set_pattern_path(pattern_path)
    my_render.set_label_path(label_path)

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
        if foggy:
            print('|_Fog simulation started.')
            my_render.add_fog(args.heavy, args.medium, args.light)
        if rainy:
            print('|_Rain simulation started.')
            my_render.add_rain()
        if smoky:
            print('|_Smoke simulation started.')
            my_render.label_parse()
            for j in range(2):
                my_render.set_smoke_color(color_list[j], color_flag[j])
                my_render.add_smoke(args.heavy, args.medium, args.light)
        end_time = time.time()
        print('|_Time elapsed {:.3f}s'.format(end_time - start_time))

    print('All finished!')
