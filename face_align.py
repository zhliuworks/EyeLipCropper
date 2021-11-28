'''
Align faces to generate 68 landmarks and bounding boxes
'''

from __future__ import absolute_import, division, print_function

import argparse
import os

import face_alignment
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='align faces with `https://github.com/1adrianb/face-alignment`')
    parser.add_argument('--images-path', type=str,
                        default='./test/images', help='the input frames path')
    parser.add_argument('--landmarks-path', type=str,
                        default='./test/landmarks', help='the output 68 landmarks path')
    parser.add_argument('--boxes-path', type=str,
                        default='./test/boxes', help='the output bounding boxes path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cpu or gpu cuda device')
    parser.add_argument('--log-path', type=str, default='./test/logs',
                        help='logging when there are no faces detected')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.landmarks_path, exist_ok=True)
    os.makedirs(args.boxes_path, exist_ok=True)
    fan = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, device=args.device, flip_input=False)
    preds = fan.get_landmarks_from_directory(
        args.images_path, return_bboxes=True)
    for image_file, (landmark, _, box) in preds.items():
        if not box:
            os.makedirs(args.log_path, exist_ok=True)
            with open(os.path.join(args.log_path, 'log.txt'), 'a') as logger:
                logger.write(os.path.abspath(image_file) + '\n')
            continue
        landmark = np.array(landmark)[0]
        box = np.array(box)[0, :4]
        npy_file_name = os.path.splitext(
            os.path.basename(image_file))[0] + '.npy'
        image_landmark_path = os.path.join(args.landmarks_path, npy_file_name)
        image_box_path = os.path.join(args.boxes_path, npy_file_name)
        np.save(image_landmark_path, landmark)
        np.save(image_box_path, box)


if __name__ == '__main__':
    main()
