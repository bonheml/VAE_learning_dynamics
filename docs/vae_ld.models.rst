##############
Models
##############

Implementation of various types of variational autoencoders (VAEs).

Learning objectives
===================
VAE learning objectives aiming to encourage disentangled representation learning.
Each of the following learning objective can be used interchangeably with different encoder/decoder type.

Generic VAE
-----------
.. autoclass:: vae_ld.models.vaes.VAE
   :members:
   :undoc-members:
   :show-inheritance:

beta-VAE
--------
.. autoclass:: vae_ld.models.vaes.BetaVAE
   :members:
   :undoc-members:
   :show-inheritance:

Annealed VAE
------------
.. autoclass:: vae_ld.models.vaes.AnnealedVAE
   :members:
   :undoc-members:
   :show-inheritance:

beta-TC VAE
-----------
.. autoclass:: vae_ld.models.vaes.BetaTCVAE
   :members:
   :undoc-members:
   :show-inheritance:

DIP VAE
-------
.. autoclass:: vae_ld.models.vaes.DIPVAE
   :members:
   :undoc-members:
   :show-inheritance:

Encoders
========
Different types of encoders that can be used with the learning objective previously described.

.. note::
   Each layer output is exposed to allow computation of representational similarity between
   every layers later on.

Fully connected encoder
-----------------------
.. autoclass:: vae_ld.models.encoders.FullyConnectedEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Convolutional encoder
---------------------
.. autoclass:: vae_ld.models.encoders.ConvolutionalEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Decoders
========
Different types of decoders that can be used with the learning objective previously described.

.. note::
   Each layer output is exposed to allow computation of representational similarity between
   every layers later on.

Fully connected decoder
-----------------------
.. autoclass:: vae_ld.models.decoders.FullyConnectedDecoder
   :members:
   :undoc-members:
   :show-inheritance:

Deconvolutional decoder
-----------------------
.. autoclass:: vae_ld.models.decoders.DeconvolutionalDecoder
   :members:
   :undoc-members:
   :show-inheritance:

Custom Layers
=============
The custom layers used by VAEs.

Sampling layer
--------------
.. autoclass:: vae_ld.models.encoders.Sampling
   :members:
   :undoc-members:
   :show-inheritance:

Losses
======
The different reconstruction losses implemented.

Bernoulli Loss
--------------
.. autoclass:: vae_ld.models.losses.BernoulliLoss
   :members:
   :undoc-members:
   :show-inheritance:

Custom Callbacks
================
The custom callbacks implemented.

Image generator
---------------
.. autoclass:: vae_ld.models.callbacks.ImageGeneratorCallback
   :members:
   :undoc-members:
   :show-inheritance:

Model utils
===========
.. automodule:: vae_ld.models.vae_utils
   :members:
   :undoc-members:
   :show-inheritance:
