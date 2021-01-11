import jax
from jax import numpy as jnp
from jax import jit
import haiku as hk

NUM_CLASSES = 10

@jit
def accuracy(logits, labels):
    """
    Args:
        logits: [batch, num_classes] float array.
        labels: categorical labels [batch,] int array (not one-hot).
    """
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds==labels)


def crossentropy_loss(logits, labels, label_smoothing=0.):
    """Compute cross entropy for logits and labels w/ label smoothing
    Args:
        logits: [batch, num_classes] float array.
        labels: categorical labels [batch,] int array (not one-hot).
        label_smoothing: label smoothing constant, used to determine the on and off values.
    """
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(labels, num_classes)
    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + label_smoothing / num_classes
    logp = jax.nn.log_softmax(logits)
    return -jnp.sum(logp * labels) # summed loss over batch 
                                   # equal to model_loglikelihood(data | params)


def log_prior(params):
    """Gaussian prior used to regularize weights (same as initialization).
    unscaled."""
    logp_tree = jax.tree_map(lambda param: - param**2 / 100**2 / 2, params)
#         lambda param: jax.scipy.stats.norm.logpdf(param, loc=0, scale=1/100),
#         params) # removed so that loss is > 0.
    return jax.tree_util.tree_reduce(lambda x, y: jnp.sum(x)+jnp.sum(y), logp_tree)

# Initialize all weights and biases the same way
initializer = hk.initializers.RandomNormal(stddev=1 / 100)

def model_fn(image):
    """returns logits"""
    image = image.astype(jnp.float32)
    convnet = hk.Sequential([
        hk.Conv2D(32, kernel_shape=(3,3), w_init=initializer, b_init=initializer),
        jax.nn.relu,
        hk.MaxPool(window_shape=(2,2), strides=2, padding="VALID"),

        hk.Conv2D(64, kernel_shape=(3,3), w_init=initializer, b_init=initializer),
        jax.nn.relu,
        hk.MaxPool(window_shape=(2,2), strides=2, padding="VALID"),

        hk.Flatten(),
        hk.Linear(NUM_CLASSES, w_init=initializer, b_init=initializer),
    ])
    return convnet(image)

model = hk.without_apply_rng(hk.transform(model_fn))
