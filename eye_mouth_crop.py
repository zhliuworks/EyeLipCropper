'''
Crop eye (left, right), mouth ROIs based on our cropper
'''

from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np
from skimage import io
from tqdm import tqdm

from cropper.eye_cropper import crop_eye_image
from cropper.mouth_cropper import crop_mouth_image


def parse_args():
    parser = argparse.ArgumentParser(description='crop eye and mouth regions')

    # common arguments
    parser.add_argument('--images-path', type=str, default='./test/images',
                        help='[COMMON] the input frames path')
    parser.add_argument('--landmarks-path', type=str, default='./test/landmarks',
                        help='[COMMON] the input 68 landmarks path')

    # eyes cropping arguments
    parser.add_argument('--boxes-path', type=str, default='./test/boxes',
                        help='[EYE] the input bounding boxes path')
    parser.add_argument('--eye-width', type=int, default=60,
                        help='[EYE] width of cropped eye ROIs')
    parser.add_argument('--eye-height', type=int, default=48,
                        help='[EYE] height of cropped eye ROIs')
    parser.add_argument('--face-roi-width', type=int, default=300,
                        help='[EYE] maximize this argument until there is a warning message')
    parser.add_argument('--face-roi-height', type=int, default=300,
                        help='[EYE] maximize this argument until there is a warning message')
    parser.add_argument('--left-eye-path', type=str, default='./test/left_eye',
                        help='[EYE] the output left eye images path')
    parser.add_argument('--right-eye-path', type=str, default='./test/right_eye',
                        help='[EYE] the output right eye images path')

    # mouth cropping arguments
    parser.add_argument('--mean-face', type=str, default='./cropper/20words_mean_face.npy',
                        help='[MOUTH] mean face pathname')
    parser.add_argument('--mouth-width', type=int, default=96,
                        help='[MOUTH] width of cropped mouth ROIs')
    parser.add_argument('--mouth-height', type=int, default=96,
                        help='[MOUTH] height of cropped mouth ROIs')
    parser.add_argument('--start-idx', type=int, default=48,
                        help='[MOUTH] start of landmark index for mouth')
    parser.add_argument('--stop-idx', type=int, default=68,
                        help='[MOUTH] end of landmark index for mouth')
    parser.add_argument('--window-margin', type=int, default=12,
                        help='[MOUTH] window margin for smoothed_landmarks')
    parser.add_argument('--mouth-path', type=str, default='./test/mouth',
                        help='[MOUTH] the output mouth images path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # crop eyes
    print('\033[36mCropping eye images ...\033[0m')
    os.makedirs(args.left_eye_path, exist_ok=True)
    os.makedirs(args.right_eye_path, exist_ok=True)
    for box_file in tqdm(sorted(os.listdir(args.boxes_path))):
        box_path = os.path.join(args.boxes_path, box_file)
        landmarks_path = os.path.join(args.landmarks_path, box_file)
        box_file = os.path.splitext(box_file)[0] + '.png'
        image_path = os.path.join(args.images_path, box_file)
        left_eye_img, right_eye_img, _, _ = crop_eye_image(np.load(landmarks_path),
                                                           np.load(box_path),
                                                           image_path,
                                                           eye_width=args.eye_width,
                                                           eye_height=args.eye_height,
                                                           face_width=args.face_roi_width,
                                                           face_height=args.face_roi_height)
        if left_eye_img is None or right_eye_img is None:
            print(f'\033[35m[WARNING] Failed to crop eye image in {box_file}, \
            please lower the argument `--face-roi-width` or `--face-roi-height`\033[0m')
        else:
            io.imsave(os.path.join(args.left_eye_path, box_file), left_eye_img)
            io.imsave(os.path.join(
                args.right_eye_path, box_file), right_eye_img)

    # crop mouth
    print('\033[36mCropping mouth images ...\033[0m')
    os.makedirs(args.mouth_path, exist_ok=True)
    crop_mouth_image(args.images_path,
                     args.landmarks_path,
                     args.mouth_path,
                     np.load(args.mean_face),
                     crop_width=args.mouth_width,
                     crop_height=args.mouth_height,
                     start_idx=args.start_idx,
                     stop_idx=args.stop_idx,
                     window_margin=args.window_margin)


if __name__ == '__main__':
    main()
