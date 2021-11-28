'''
Crop mouth ROIs
adapted from https://github.com/ahaliassos/LipForensics/tree/main/preprocessing
'''

from __future__ import absolute_import, division, print_function

import os
from collections import deque

import numpy as np
from PIL import Image
from skimage import transform as tf
from tqdm import tqdm


def warp_img(src, dst, img, std_size):
    ''' '
    Warp image to match mean face landmarks
    Parameters
    ----------
    src : numpy.array
        Key Landmarks of initial face
    dst : numpy.array
        Key landmarks of mean face
    img : numpy.array
        Frame to be aligned
    std_size : tuple
        Target size for frames
    '''
    tform = tf.estimate_transform(
        'similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse,
                     output_shape=std_size)  # wrap the frame image
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    ''' '
    Apply affine transform to image.
    Parameters
    ----------
    transform : skimage.transform._geometric.GeometricTransform
        Object with transformation parameters
    img : numpy.array
        Frame to be aligned
    std_size : tuple
        Target size for frames
    '''
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    ''' '
    Crop square mouth region given landmarks
    Parameters
    ----------
    img : numpy.array
        Frame to be cropped
    landmarks : numpy.array
        Landmarks corresponding to mouth region
    height : int
        Height of output image
    width : int
        Width of output image
    threshold : int, optional
        Threshold for determining whether to throw an exception when the initial bounding box is out of bounds
    '''
    center_x, center_y = np.mean(landmarks, axis=0)
    if center_y - height < 0:
        center_y = height
    if int(center_y) - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')
    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    img_cropped = np.copy(
        img[
            int(round(center_y) - round(height)): int(round(center_y) + round(height)),
            int(round(center_x) - round(width)): int(round(center_x) + round(width)),
        ]
    )
    return img_cropped


def crop_mouth_image(video_path, landmarks_dir, target_dir, mean_face_landmarks,
                     crop_width=96, crop_height=96, start_idx=48, stop_idx=68, window_margin=12):
    '''
    Align frames and crop mouths. The landmarks are smoothed over 12 frames to account for motion jitter, and each frame
    is affine warped to the mean face via five landmarks (around the eyes and nose). The mouth is cropped in each frame
    by resizing the image and then extracting a fixed 96 by 96 region centred around the mean mouth landmark.

    Parameters
    ----------
    video_path : str
        Path to video directory containing frames of faces
    landmarks_dir : str
        Path to directory of landmarks for each frame
    target_dir : str
        Path to target directory for cropped frames
    mean_face_landmarks : numpy.array
        Landmarks for the mean face of a dataset (in this case, the LRW dataset)
    crop_width : int
        Width of mouth ROIs
    crop_height : int
        Height of mouth ROIs
    start_idx : int
        Start of landmark index for mouth
    stop_idx : int
        End of landmark index for mouth
    window_margin : int
        Window margin for smoothed_landmarks
    '''
    STD_SIZE = (256, 256)
    STABLE_POINTS = [33, 36, 39, 42, 45]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    frame_names = sorted(os.listdir(video_path))

    q_frames, q_landmarks, q_name = deque(), deque(), deque()
    for frame_name in tqdm(frame_names):
        landmark_path = os.path.join(landmarks_dir, f'{frame_name[:-4]}.npy')
        if os.path.exists(landmark_path):
            landmarks = np.load(landmark_path)
        else:
            continue

        with Image.open(os.path.join(video_path, frame_name)) as pil_img:
            img = np.array(pil_img)

        # Add elements to the queues
        q_frames.append(img)
        q_landmarks.append(landmarks)
        q_name.append(frame_name)

        if len(q_frames) == window_margin:  # Wait until queues are large enough
            smoothed_landmarks = np.mean(q_landmarks, axis=0)

            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frames.popleft()
            cur_name = q_name.popleft()

            # Get aligned frame as well as affine transformation that produced it
            trans_frame, trans = warp_img(
                smoothed_landmarks[STABLE_POINTS,
                                   :], mean_face_landmarks[STABLE_POINTS, :], cur_frame, STD_SIZE
            )

            # Apply that affine transform to the landmarks
            trans_landmarks = trans(cur_landmarks)

            # Crop mouth region
            cropped_frame = cut_patch(
                trans_frame,
                trans_landmarks[start_idx: stop_idx],
                crop_height // 2,
                crop_width // 2,
            )

            # Save image
            target_path = os.path.join(target_dir, cur_name)
            Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)

    # Process remaining frames in the queue
    while q_frames:
        cur_frame = q_frames.popleft()
        cur_name = q_name.popleft()
        cur_landmarks = q_landmarks.popleft()

        trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
        trans_landmarks = trans(cur_landmarks)

        cropped_frame = cut_patch(
            trans_frame, trans_landmarks[start_idx:
                                         stop_idx], crop_height // 2, crop_width // 2
        )

        target_path = os.path.join(target_dir, cur_name)
        Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)
