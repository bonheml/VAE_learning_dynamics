from distutils.extension import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools import dist

# Allows np to be downloaded before import
dist.Distribution().fetch_build_eggs(["numpy>=1.19"])

import numpy as np

hidalgo = Extension("gibbs", sources=["vae_ld/ext/gibbs.c"], include_dirs=[np.get_include()])

extras = {
   'doc': ['sphinx==5.0.2', 'sphinx-rtd-theme']
}

setup(
    name="vae_ld",
    version="0.1",
    description="Library for research on VAE learning dynamics",
    author="Lisa Bonheme",
    author_email="lb732@kent.ac.uk",
    url="https://github.com/bonheml/VAE_learning_dynamics",
    license="Apache 2.0",
    ext_modules=[hidalgo],
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        "bin/train",
        "bin/visualise_similarity",
        "bin/visualise_images",
        "bin/get_data_ide",
        "bin/hidalgo",
        "bin/compute_similarity",
        "bin/compute_ph",
        "bin/test_dataset",
        "bin/filter_variables",
        "bin/stitch_train",
        "bin/get_optimal_nb_latents",
        "bin/visualise_ides",
        "bin/latent_traversal",
    ],
    install_requires=[
        "pillow>=7.2.0",
        "pandas>=1.0.5",
        "tensorflow_hub~=0.12",
        "tensorflow~=2.6",
        "tensorboard~=2.6",
        "imageio",
        "hydra-core",
        "scikit-learn",
        "pandas",
        "simplejson",
        "six",
        "requests",
        "matplotlib",
        "seaborn",
        "tensorflow_probability",
        "scipy",
        "gudhi",
        "jax[cpu]"
    ],
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="tensorflow2, machine learning, variational autoencoders, deep learning, representational similarity",
)
