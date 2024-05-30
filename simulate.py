import os
import time
import argparse

import render


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foggy', default=False, action='store_true')
    parser.add_argument('--rainy', default=False, action='store_true')
    parser.add_argument('--smoky', default=False, action='store_true')
    parser.add_argument('--cloudy', default=False, action='store_true')
    parser.add_argument('--light', default=False, action='store_true')
    parser.add_argument('--medium', default=True, action='store_true')
    parser.add_argument('--heavy', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    path = 'data'
    foggy = args.foggy
    rainy = args.rainy
    smoky = args.smoky
    cloudy = args.cloudy

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
    cloud_path = os.path.join(path, 'cloudy')
    if not os.path.exists(cloud_path):
        os.mkdir(cloud_path)

    label_path = os.path.join(path, 'labels')
    pattern_path = os.path.join(path, 'patterns')

    image_list = os.listdir(raw_path)

    print('Detect', colorstr('{}'.format(len(image_list))), 'files.')

    my_render = render.Render()
    my_render.set_raw_path(raw_path)
    my_render.set_fog_path(fog_path)
    my_render.set_rain_path(rain_path)
    my_render.set_smoke_path(smoke_path)
    my_render.set_cloud_path(cloud_path)
    my_render.set_mask_path(mask_path)
    my_render.set_pattern_path(pattern_path)
    my_render.set_label_path(label_path)

    if args.light:
        my_render.light = True
    if args.medium:
        my_render.medium = True
    if args.heavy:
        my_render.heavy = True

    color_list = [240, 30]
    color_flag = ['white', 'black']

    for i in range(len(image_list)):
        if image_list[i].startswith('.'):
            print('Detect a system generated file, pass.')
            continue
        print(colorstr('[{}/{}]'.format(i + 1, len(image_list))),
              'Now processing {}.'.format(image_list[i].split('\\')[-1]))
        start_time = time.time()
        image_path = os.path.join(raw_path, image_list[i])
        my_render.read_image(image_path)
        if foggy:
            print('|_Fog simulation started.')
            my_render.add_fog()
        if rainy:
            print('|_Rain simulation started.')
            my_render.add_rain()
        if smoky:
            print('|_Smoke simulation started.')
            my_render.label_parse()
            for j in range(2):
                my_render.set_smoke_color(color_list[j], color_flag[j])
                my_render.add_smoke()
        if cloudy:
            print('|_Cloud simulation started.')
            my_render.set_smoke_color(color_list[0], color_flag[0])
            my_render.add_cloud()
        end_time = time.time()
        print('|_Time elapsed {:.3f}s'.format(end_time - start_time))

    print(colorstr('All finished!'))
