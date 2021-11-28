'''
Extract frames of the given video
'''

from __future__ import absolute_import, division, print_function

import argparse
import os

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='extract frames with opencv')
    parser.add_argument('--video-path', type=str,
                        default='./test/video.mp4', help='the input video path')
    parser.add_argument('--images-path', type=str,
                        default='./test/images', help='the output frames path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.images_path, exist_ok=True)
    reader = cv2.VideoCapture(args.video_path)
    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        cv2.imwrite(os.path.join(args.images_path,
                    '{:04d}.png'.format(frame_num)), image)
        frame_num += 1
    reader.release()


if __name__ == '__main__':
    main()
