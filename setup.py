from setuptools import setup

setup(
    name="nvgd",
    version="0.1",
    description="Neural Variational Gradient Descent",
    author="Lauro Langosco",
    author_email="langosco.lauro@gmail.com",
    packages=["nvgd"],
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "tqdm",
        "argparse",
        "matplotlib",
        "chex==0.0.2",
        "dm-haiku==0.0.3",
        "dm-tree==0.1.5",
        "optax==0.0.1",
        "json-tricks==3.15.2",
        "dataclasses>=0.6",
        "pandas",
        "scikit-learn",
        "scipy",
        "POT",
        "tensorflow-probability[jax]",
    ]
)
