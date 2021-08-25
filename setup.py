from setuptools import find_packages
from setuptools import setup

setup(
    name="learning_dynamics_vae",
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
        "bin/visualise_learning_dynamics"
    ],
    install_requires=[
        "imageio",
        "hydra-core",
        "scikit-learn",
        "numpy",
        "pandas",
        "simplejson",
        "six",
        "requests",
        "matplotlib>=3.3.0",
        "seaborn",
        "pillow>=7.2.0",
        "pandas>=1.0.5",
        "scipy==1.4.1",
        "tensorflow_hub>=0.8.0",
        "tensorflow_probability==0.10.1",
        "tensorflow==2.5.1",
        "tensorboard==2.2.2"
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="tensorflow2, machine learning, variational auto-encoders, deep learning",
)