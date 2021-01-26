import jax
from jax import numpy as jnp
from jax import jit, vmap
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
    return jnp.mean(preds == labels)


@jit
def ensemble_accuracy(logits, labels):
    """use ensemble predictions to compute validation accuracy.
    args:
        logits: result from vmap(model.apply, (0, None))(param_set, images)
        labels: batch of corresponding labels, shape (batch,)"""
    preds = jnp.mean(vmap(jax.nn.softmax)(logits), axis=0)  # mean prediction
    return jnp.mean(preds.argmax(axis=1) == labels)


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
    return -jnp.sum(logp * labels)  # summed loss over batch
                                    # equal to model_loglikelihood(data | params)


def log_prior(params):
    """Gaussian prior used to regularize weights (same as initialization).
    unscaled."""
#         lambda param: jax.scipy.stats.norm.logpdf(param, loc=0, scale=1/100),
#         params) # removed so that loss is > 0.
    params_flat, _ = jax.flatten_util.ravel_pytree(params)
    return - jnp.sum(params_flat**2) * 100**2 / 2


# Initialize all weights and biases the same way
initializer = hk.initializers.RandomNormal(stddev=1 / 100)


def make_model(size: str = "large"):
    def model_fn(image):
        """returns logits"""
        if size == "large":
            conv = hk.Conv2D(32, kernel_shape=(3, 3), w_init=initializer, b_init=initializer)
            final_pool = lambda x: x
        elif size == "small":
            conv = hk.Conv2D(2, kernel_shape=(3, 3), w_init=initializer, b_init=initializer)
            final_pool = hk.AvgPool(window_shape=(10,), strides=(10,), padding="VALID")
        else:
            raise ValueError(f"Size must be 'large' or 'small'; received {size} instead.")

        image = image.astype(jnp.float32)
        convnet = hk.Sequential([
            conv,
            jax.nn.relu,
            hk.MaxPool(window_shape=(2, 2), strides=2, padding="VALID"),

            conv,
            jax.nn.relu,
            hk.MaxPool(window_shape=(2, 2), strides=2, padding="VALID"),

            hk.Flatten(),
            final_pool,
            hk.Linear(NUM_CLASSES, w_init=initializer, b_init=initializer),
        ])
        return convnet(image)
    return hk.without_apply_rng(hk.transform(model_fn))
