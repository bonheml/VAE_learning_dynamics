################
Getting started
################

Installation
============
After downloading the source code, in the main folder, do

.. code-block::

   $ pip install -e .

This will install an editable version of the package.

You will then need to export two environment variable:
    * `XP_PATH` : the absolute path to the folder used to store the experiment results
    * `DATA_PATH` : the absolute path to the folder used to store the data

.. Warning::
   When defining these two variables, make sure that you choose an existing folder.

For example,

.. code-block::

   $ export XP_PATH=/home/username/experiment/results
   $ export DATA_PATH=/home/username/data

A first example
===============

Once the installation is done, some command-line tools are directly accessible:
   * `train` which trains a VAE
   * `compute_similarity` which computes similarity scores of trained models
   * `visualise_similarity` which generates figures from similarity scores

They all use the configuration framework `Hydra <https://hydra.cc/>`_.
Thus, as usual for Hydra-based scripts, optional arguments for each command can be displayed using the `--help` option.
For example,

.. code-block::

    $ visualise_similarity --help
	visualise_similarity is powered by Hydra.

	== Configuration groups ==
	Compose your configuration from those groups (group=option)

	callbacks: early_stopping, image_generator, model_checkpoint, tensorboard
	dataset: cars, color_dsprites, dsprites, grey_dsprites, mnist, noisy_dsprites, scream_dsprites, shapes3d, smallnorb
	dimensionality_reduction: PCA
	model: annealed_vae, beta_tc_vae, beta_vae, classifier, dip_vae_i, dip_vae_ii, factor_vae, gon
	model/clf: convolutional, fully_connected
	model/decoder: convolutional, dummy, fully_connected, stitched
	model/encoder: convolutional, dummy, fully_connected, gon, stitched
	model/reconstruction_loss_fn: bernoulli
	optimizer: adam
	sampling: sampler
	similarity_metric: cka, procrustes
	visualisation_tool: aggregate, heatmap, layer_list, layer_pair, tsne


	== Config ==
	Override anything in the config (foo.bar=value)

	dataset_name: cars3d
	m1_name: beta_vae
	metric_name: cka
	overwrite: false
	latent_dim: 10
	visualisation_tool:
	    name: heatmap
	    fn:
		    _target_: vae_ld.visualisation.similarity.similarity_heatmap
		    metric_name: ${metric_name}
		    input_file: ${oc.env:XP_PATH}/visualise_similarity/latent_${latent_dim}/aggregate/${dataset_name}/${metric_name}/${m1_name}/${metric_name}_${dataset_name}_${m1_name}_agg.tsv
		    overwrite: ${overwrite}

	Powered by Hydra (https://hydra.cc)
	Use --hydra-help to view Hydra specific help

You can start to use them as-is.
