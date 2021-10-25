from setuptools import find_packages
from setuptools import setup

setup(
    name="vae_ld",
    version="0.1",
    description="Library for research on VAE learning dynamics",
    author="Lisa Bonheme",
    author_email="lb732@kent.ac.uk",
    url="https://github.com/bonheml/VAE_learning_dynamics",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        "bin/train",
        "bin/visualise",
        "bin/get_data_ide",
        "bin/compute_cka",
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
        "numpy",
        "pandas",
        "simplejson",
        "six",
        "requests",
        "matplotlib",
        "seaborn",
        "tensorflow_probability",
        "scipy"
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="tensorflow2, machine learning, variational auto-encoders, deep learning",
)