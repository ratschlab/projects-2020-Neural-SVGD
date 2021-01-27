from itertools import cycle

import config as cfg
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import numpy as onp

print("Loading data...")
# Load MNIST
mnist_data, info = tfds.load(name="mnist",
                             batch_size=-1,
                             data_dir=cfg.data_dir,
                             with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']

# Full train and test set
train_images, train_labels = train_data['image'], train_data['label']
test_images, test_labels = test_data['image'], test_data['label']

# Split off the validation set
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=0)

train_data_size = len(train_images)
steps_per_epoch = train_data_size // cfg.batch_size


def _make_batches(images, labels, batch_size, cyclic=True):
    """Returns an iterator through tuples (image_batch, label_batch).
    if cyclic, then the iterator cycles back after exhausting the batches"""
    num_batches = len(images) // batch_size
    split_idx = onp.arange(1, num_batches+1)*batch_size
    batches = zip(*[onp.split(data, split_idx, axis=0) for data in (images, labels)])
    return cycle(batches) if cyclic else list(batches)


def make_batches(batch_size):
    validation_batches = _make_batches(
        val_images, val_labels, 1024, cyclic=False)
    training_batches = _make_batches(train_images, train_labels, batch_size)
    test_batches = _make_batches(train_images, train_labels, batch_size)
    return (training_batches, validation_batches, test_batches)


training_batches, validation_batches, test_batches = make_batches(cfg.batch_size)
