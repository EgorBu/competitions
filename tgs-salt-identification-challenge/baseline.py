import argparse
import logging

from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from simple_reg_model import build_cnn
from prepare_data import load_all
from utilities import find_threshold, RLenc


def train_pipeline(X_train, y_train, X_valid, y_valid, epochs=100, batch_size=32,
                   patience=5, model_save_loc="model.hdf5"):
    early_stop = EarlyStopping(patience=patience, verbose=1, monitor="mean_iou", mode="max")
    check_point = ModelCheckpoint(model_save_loc, save_best_only=True, verbose=1,
                                  monitor="mean_iou", mode="max")

    model = build_cnn()
    print(model.summary())

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid),
              callbacks=[early_stop, check_point], batch_size=batch_size)

    return model, model.evaluate(X_valid, y_valid, batch_size=batch_size)


def baseline(train_image_dir, train_mask_dir, test_image_dir, depth_loc, nfolds=5,
             output="output/submission2.csv"):
    # load data
    X_tr, y_tr, z_tr, X_t, z_t, test_files = load_all(
        train_image_dir, train_mask_dir, test_image_dir, depth_loc
    )

    # k-folds + training
    test_predictions = None  # to save predictions from different models
    train_predictions = None  # to save predictions from different models
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(kf.split(X_tr)):
        print("train model {} out of {}...".format(i + 1, nfolds))

        model, results = train_pipeline(X_tr[train_index], y_tr[train_index], X_tr[val_index],
                                        y_tr[val_index])
        print("validation results", results)

        if test_predictions is None:
            test_predictions = model.predict(X_t)
        else:
            test_predictions += model.predict(X_t)

        if train_predictions is None:
            train_predictions = model.predict(X_tr)
        else:
            train_predictions += model.predict(X_tr)
        del model
    train_predictions = train_predictions / (i + 1)
    test_predictions = test_predictions / (i + 1)

    # binarization of matrix
    threshold = find_threshold(y_tr, train_predictions)
    prediction_to_submit = test_predictions > threshold

    pred_dict = {fn: RLenc(np.round(prediction_to_submit[i, :, :, 0]))
                 for i, fn in tqdm(enumerate(test_files))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(output)


def get_parser() -> argparse.ArgumentParser:
    """
    Creates the cmdline argument parser.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")
    # , , test_image_dir, depth_loc, nfolds = 5,
    # output_loc = "output/submission.csv"
    parser.add_argument("--train-image-dir", required=True, help="Directory with train images.")
    parser.add_argument("--train-mask-dir", required=True, help="Directory with train masks.")
    parser.add_argument("--test-image-dir", required=True, help="Directory with test images.")
    parser.add_argument("--depth-loc", required=True, help="Path to file with depth.")
    parser.add_argument("--nfolds", default=5, type=int, help="Number of folds to use.")
    parser.add_argument("-o", "--output", required=True, help="Path to save predictions.")

    # TODO: add functionality to save averaged predictions from the models

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    baseline(train_image_dir=args.train_image_dir, train_mask_dir=args.train_mask_dir,
             test_image_dir=args.test_image_dir, depth_loc=args.depth_loc, nfolds=args.nfolds,
             output=args.output)


if __name__ == "__main__":
    main()
