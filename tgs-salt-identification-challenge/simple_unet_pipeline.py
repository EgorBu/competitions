import argparse
import logging

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from unet_second import build_unet
from prepare_data import load_all
from utilities import find_threshold, RLenc, set_random_seed, config_keras


def train_pipeline(X_train, y_train, X_valid, y_valid, epochs=200, batch_size=96,
                   patience=10, model_save_loc="model_simple.hdf5", monitor="val_loss",
                   mode="min", factor=0.1, lr_patience=5, min_lr=0.00001):
    # Модели не используются сингл для финального решения. А в ансамбле бывает, что лучше юзать
    # модель с низким лоссом или просто обученную по-другому для вариативности.
    # monitor = "val_loss" mode = "min"
    early_stop = EarlyStopping(patience=patience, verbose=1, monitor=monitor, mode=mode)
    check_point = ModelCheckpoint(model_save_loc, save_best_only=True, verbose=1,
                                  monitor=monitor, mode=mode)
    reduce_lr = ReduceLROnPlateau(factor=factor, patience=lr_patience, min_lr=min_lr, verbose=1,
                                  monitor=monitor, mode=mode)

    model = build_unet()
    print(model.summary())

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid),
              callbacks=[early_stop, check_point, reduce_lr], batch_size=batch_size)

    return model, model.evaluate(X_valid, y_valid, batch_size=batch_size)


def baseline(train_image_dir, train_mask_dir, test_image_dir, depth_loc, nfolds=5,
             output="output/submission.csv"):
    set_random_seed()
    config_keras()
    model_save_loc = "model_almost_unet.hdf5"
    # load data
    X_tr, y_tr, z_tr, X_t, z_t, test_files = load_all(
        train_image_dir, train_mask_dir, test_image_dir, depth_loc
    )

    # flip images
    X_tr = np.append(X_tr, [np.fliplr(x) for x in X_tr], axis=0)
    y_tr = np.append(y_tr, [np.fliplr(x) for x in y_tr], axis=0)
    # X_tr = np.append(X_tr, [np.flipud(x) for x in X_tr], axis=0)
    # y_tr = np.append(y_tr, [np.flipud(x) for x in y_tr], axis=0)

    # k-folds + training
    test_predictions = None  # to save predictions from different models
    train_predictions = None  # to save predictions from different models
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(kf.split(X_tr)):
        print("train model {} out of {}...".format(i + 1, nfolds))

        model, results = train_pipeline(X_tr[train_index], y_tr[train_index], X_tr[val_index],
                                        y_tr[val_index], model_save_loc=model_save_loc)
        model.load_weights(model_save_loc, by_name=False)
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
        break

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
