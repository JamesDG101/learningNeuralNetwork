import os
from zipfile import BadZipFile
import numpy as np
from emnist import clear_cached_data, extract_test_samples, extract_training_samples
from scipy.io import loadmat

def load_emnist_from_mat(mat_path='emnist-letters.mat'):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Fallback file not found: {mat_path}")

    dataset = loadmat(mat_path)['dataset'][0, 0]
    train = dataset['train'][0, 0]
    test = dataset['test'][0, 0]

    train_images = train['images'].reshape((-1, 28, 28)).swapaxes(1, 2)
    train_labels = train['labels'].ravel()
    test_images = test['images'].reshape((-1, 28, 28)).swapaxes(1, 2)
    test_labels = test['labels'].ravel()

    return train_images, train_labels, test_images, test_labels


def load_emnist():
    try:
        train_images, train_labels = extract_training_samples('letters')
        test_images, test_labels = extract_test_samples('letters')
    except BadZipFile:
        clear_cached_data()
        try:
            train_images, train_labels = extract_training_samples('letters')
            test_images, test_labels = extract_test_samples('letters')
        except BadZipFile:
            train_images, train_labels, test_images, test_labels = load_emnist_from_mat()

    return train_images, train_labels.ravel(), test_images, test_labels.ravel()


def split_training_validation(images, labels, validation_ratio=0.1, seed=42):
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    val_count = int(len(images) * validation_ratio)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    x_train = images[train_indices]
    y_train = labels[train_indices]
    x_val = images[val_indices]
    y_val = labels[val_indices]

    return x_train, y_train, x_val, y_val


def preprocess_images(images, normalize=True, flatten=False):
    if images.shape[1:] != (28, 28):
        raise ValueError(f"Expected images of shape (N, 28, 28), got {images.shape}")

    processed = images.astype(np.float32)
    if normalize:
        processed = processed / 255.0

    if flatten:
        processed = processed.reshape(processed.shape[0], 28 * 28)

    return processed


def one_hot_encode(labels, num_classes=26):
    if np.min(labels) < 0 or np.max(labels) >= num_classes:
        raise ValueError("Label values are out of range for one-hot encoding")
    return np.eye(num_classes, dtype=np.float32)[labels]


def prepare_emnist_data(validation_ratio=0.1, seed=42, normalize=True, flatten=True):
    train_images, train_labels, test_images, test_labels = load_emnist()

    train_labels = train_labels - 1
    test_labels = test_labels - 1

    x_train_raw, y_train, x_val_raw, y_val = split_training_validation(
        train_images,
        train_labels,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    x_train = preprocess_images(x_train_raw, normalize=normalize, flatten=flatten)
    x_val = preprocess_images(x_val_raw, normalize=normalize, flatten=flatten)
    x_test = preprocess_images(test_images, normalize=normalize, flatten=flatten)

    y_train_onehot = one_hot_encode(y_train, num_classes=26)
    y_val_onehot = one_hot_encode(y_val, num_classes=26)
    y_test_onehot = one_hot_encode(test_labels, num_classes=26)

    return {
        'x_train': x_train,
        'y_train': y_train,
        'y_train_onehot': y_train_onehot,
        'x_val': x_val,
        'y_val': y_val,
        'y_val_onehot': y_val_onehot,
        'x_test': x_test,
        'y_test': test_labels,
        'y_test_onehot': y_test_onehot,
    }

if __name__ == "__main__":
    data = prepare_emnist_data(
        validation_ratio=0.1,
        seed=42,
        normalize=True,
        flatten=True,
    )
