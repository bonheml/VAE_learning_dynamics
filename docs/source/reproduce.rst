#######################
Reproducing experiments
#######################

.. Warning::
   Make sure you follow the :ref:`Installation` steps before running commands form this section.

Reproduce the results of *How do Variational Autoencoders Learn? Insights from Representational Similarity*
===========================================================================================================
The experiment from `this paper <paper_arxiv_url>`_ was divided into three steps:
    * :ref:`Training VAEs`
    * :ref:`Computing similarity scores`
    * :ref:`Visualising the results`


Training VAEs
-------------
.. Note::
   The pre-trained models of the papers can be `downloaded <data_url>`_ if you want to skip this step.
   When downloading this, make sure to put them in the directory corresponding to your `XP_PATH` env variable without
   modifying the folder architecture used.

Model can be retrained via the command line tool.
For example,

.. code-block::

   $ train dataset=cars model=beta_vae param_value=2 seed=0

will train a beta-VAE on Cars3D dataset with beta of 2 and a seed of 0.

Multi-run can be performed with the `-m` option:

.. code-block::

   $ train -m dataset=cars model=beta_vae param_value=2 seed=0,1,2,3,4

This will train beta-VAE on Cars3D dataset with beta of 2 using seeds 0 to 4.

The complete set of models trained for this paper could be retrained with

.. code-block::

   $ train -m dataset=cars,dsprites,smallnorb model=beta_vae param_value=1,2,4,6,8 seed=0,1,2,3,4
   $ train -m dataset=cars,dsprites,smallnorb model=beta_tc_vae param_value=1,2,4,6,8 seed=0,1,2,3,4
   $ train -m dataset=cars,dsprites,smallnorb model=annealed_vae param_value=5,10,25,50,75 seed=0,1,2,3,4
   $ train -m dataset=cars,dsprites,smallnorb model=dip_vae_ii param_value=1,2,5,10,20 seed=0,1,2,3,4

This can also be done using the corresponding individual commands.


Computing similarity scores
---------------------------
.. Note::
   The aggregated similarity scores of the papers can be `downloaded <scores_url>`_ if you want to skip this step.
   When downloading this, make sure to put them in the directory corresponding to your `XP_PATH` env variable without modifying the folder architecture used.

.. Warning::
   If you do not used the available similarity scores, make sure that you have the pre-trained models before doing the following steps.

Similarity scores between two trained models can be computed via the command line tool.
For example, to compute the CKA self-similarity of a beta-VAE trained on cars3D with a beta of 2 and seed 0 taken at 5 different epochs (every 6000 steps):

.. code-block::

   $ compute_similarity similarity_metric=cka dataset=cars m1_name=beta_vae p1_value=2 m1_seed=0


One can overwrite any results previously computed by adding `overwrite=true` to the command line.
For example, to overwrite the scores computed above:

.. code-block::

   $ compute_similarity similarity_metric=cka dataset=cars m1_name=beta_vae p1_value=2 m1_seed=0 overwrite=true


Multi-run can be performed with the `-m` option.
For example to compute the CKA self-similarity of every beta-VAE trained on cars3D with a beta of 2:

.. code-block::

   $ compute_similarity -m similarity_metric=cka dataset=cars m1_name=beta_vae p1_value=2 m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4


To compute all the similarity scores between beta-VAE models one can run:

.. code-block::

   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=beta_vae p1_value=1,2,4,6,8 p2_value=1,2,4,6,8 m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4
   $ compute_similarity -m similarity_metric=procrustes similarity_metric.use_gpu=true dataset=cars,dsprites,smallnorb m1_name=beta_vae p1_value=1,2,4,6,8 p2_value=1,2,4,6,8 m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4

The first line compute all the CKA scores on CPU while the second line compute all the Procrustes scores on GPU.

.. Note::
   Procrustes is very long to compute and we strongly advise using GPU instead of CPU to reduce the computation time.

The complete set of CKA scores computed for this paper could be recomputed with:

.. code-block::

   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=beta_vae p1_value=1,2,4,6,8 p2_value=1,2,4,6,8 m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4
   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=beta_vae m2_name=beta_tc_vae p1_value=1,2,4,6,8 p2_value=1,2,4,6,8 m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4
   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=beta_vae m2_name=annealed_vae p1_value=1,2,4,6,8 p2_value=5,10,25,50,75 p2_name=c_max m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4
   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=beta_vae m2_name=dip_vae_ii p1_value=1,2,4,6,8 p2_value=1,2,5,10,20 p2_name=lambda m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4

   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=beta_tc_vae m2_name=beta_tc_vae p1_value=1,2,4,6,8 p2_value=1,2,4,6,8 m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4
   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=beta_tc_vae m2_name=annealed_vae p1_value=1,2,4,6,8 p2_value=5,10,25,50,75 p2_name=c_max m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4
   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=beta_tc_vae m2_name=dip_vae_ii p1_value=1,2,4,6,8 p2_value=1,2,5,10,20 p2_name=lambda m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4

   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=dip_vae_ii p1_value=1,2,5,10,20 p2_value=1,2,5,10,20 p1_name=lambda m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4
   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=dip_vae_ii m2_name=annealed_vae p1_value=1,2,5,10,20 p2_value=5,10,25,50,75 p1_name=lambda p2_name=c_max m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4

   $ compute_similarity -m similarity_metric=cka dataset=cars,dsprites,smallnorb m1_name=annnealed_vae p1_value=5,10,25,50,75 p2_value=5,10,25,50,75 p2_name=c_max m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4

This can also be done using the corresponding individual commands.

The complete set of Procrustes scores computed for this paper could be recomputed with similar commands, using
`similarity_metric=procrustes similarity_metric.use_gpu=true` instead of `similarity_metric=cka` and `dataset=cars`.
For example, the first line of the example above for Procrustes would be:

.. code-block::

   $ compute_similarity -m similarity_metric=procrustes similarity_metric.use_gpu=true dataset=cars m1_name=beta_vae p1_value=1,2,4,6,8 p2_value=1,2,4,6,8 m1_seed=0,1,2,3,4 m2_seed=0,1,2,3,4

Finally, the results are aggregated using:

.. code-block::

   $ visualise_similarity -m visualisation_tool=aggregate m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae metric_name=cka,procrustes dataset_name=cars3d,dsprites,smallnorb
   $ visualise_similarity -m metric_name=procrustes visualisation_tool=aggregate m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae metric_name=cka,procrustes dataset_name=cars3d

Here also, one can overwrite any results previously computed by adding `overwrite=true` to the command line.


Visualising the results
-----------------------
.. Warning::
   Make sure that you have the aggregated scores before doing the following steps.

The following (or equivalent individual runs) is computing t-SNE visualisations of CKA and Procrustes scores for all the models, datasets, and hyperparameters used:

.. code-block::

   $ visualise_similarity -m visualisation_tool=tsne m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae visualisation_tool.fn.target=seed dataset_name=cars3d,dsprites,smallnorb
   $ visualise_similarity -m metric_name=procrustes visualisation_tool=tsne m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae visualisation_tool.fn.target=seed

   $ visualise_similarity -m visualisation_tool=tsne m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae visualisation_tool.fn.target=regularisation dataset_name=cars3d,dsprites,smallnorb
   $ visualise_similarity -m metric_name=procrustes visualisation_tool=tsne m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae visualisation_tool.fn.target=regularisation

Heatmap for CKA and procrustes can be generated with:

.. code-block::

   $ visualise_similarity -m visualisation_tool=heatmap m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae dataset_name=cars3d,dsprites,smallnorb
   $ visualise_similarity -m metric_name=procrustes visualisation_tool=heatmap m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae

Lineplots visualisations are generated with:

.. code-block::

    $ visualise_similarity -m visualisation_tool=layer_pair visualisation_tool.fn.m1_layer=input,encoder/z_mean m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae dataset_name=cars3d,dsprites,smallnorb
    $ visualise_similarity -m metric_name=procrustes visualisation_tool=layer_pair visualisation_tool.fn.m1_layer=input,encoder/z_mean m1_name=beta_vae,beta_tc_vae,dip_vae_ii,annealed_vae

    $ visualise_similarity -m visualisation_tool=layer_list visualisation_tool.fn.regularisation=1,2,4,6,8 m1_name=beta_vae,beta_tc_vae dataset_name=cars3d,dsprites,smallnorb
    $ visualise_similarity -m metric_name=procrustes visualisation_tool=layer_list visualisation_tool.fn.regularisation=1,2,4,6,8 m1_name=beta_vae,beta_tc_vae
    $ visualise_similarity -m visualisation_tool=layer_list visualisation_tool.fn.regularisation=5,10,25,50,75 m1_name=annealed_vae dataset_name=cars3d,dsprites,smallnorb
    $ visualise_similarity -m metric_name=procrustes visualisation_tool=layer_list visualisation_tool.fn.regularisation=5,10,25,50,75 m1_name=annealed_vae
    $ visualise_similarity -m visualisation_tool=layer_list visualisation_tool.fn.regularisation=1,2,5,10,20 m1_name=dip_vae_ii dataset_name=cars3d,dsprites,smallnorb
    $ visualise_similarity -m metric_name=procrustes visualisation_tool=layer_list visualisation_tool.fn.regularisation=1,2,5,10,20 m1_name=dip_vae_ii
