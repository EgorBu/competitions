import os
import logging as log

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


def load_images(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    X = [np.array(cv2.imread(os.path.join(directory, p), cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
         for p in tqdm(files)]
    X = np.array(X) / 255  # scale to 0~1 interval
    return np.expand_dims(X, axis=3), files


def load_masks(directory, files):
    X = [np.array(cv2.imread(os.path    .join(directory, p), cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
         for p in tqdm(files)]
    X = np.array(X)
    X = (X > 0).astype(np.float32)  # binarization mask
    return np.expand_dims(X, axis=3), files


def load_train_data(image_dir, mask_dir, depths):
    train_images, files = load_images(image_dir)
    log.info("Number of train images:", len(files))
    train_z = np.array([depths[file.split(".")[0]] for file in files])
    train_masks, _ = load_masks(mask_dir, files)
    assert train_masks.shape[0] == train_images.shape[0]
    return train_images, train_masks, train_z


def load_test_data(image_dir, depths):
    test_images, files = load_images(image_dir)
    test_z = np.array([depths[file.split(".")[0]] for file in files])
    return test_images, test_z, [f.split(".")[0] for f in files]


def load_depths(depth_loc):
    depths = pd.read_csv(depth_loc)
    depths.columns = ["id", "z"]
    return dict((id_, z_) for id_, z_ in zip(depths["id"].tolist(), depths["z"].tolist()))


def load_all(train_image_dir, train_mask_dir, test_image_dir, depth_loc):
    depths = load_depths(depth_loc)
    X_tr, y_tr, z_tr = load_train_data(train_image_dir, train_mask_dir, depths)
    X_t, z_t, test_files = load_test_data(test_image_dir, depths)
    return X_tr, y_tr, z_tr, X_t, z_t, test_files
