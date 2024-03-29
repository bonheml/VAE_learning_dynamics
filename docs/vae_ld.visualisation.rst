#####################
Visualisation
#####################

Visualisation tools used to monitor the representational similarity.

Visualise similarity
--------------------

.. warning::
   These functions relies on the layer names of the current implementation and are not compatible with custom models.


.. autofunction:: vae_ld.visualisation.similarity.similarity_heatmap

.. autofunction:: vae_ld.visualisation.similarity.avg_similarity_layer_pair

.. autofunction:: vae_ld.visualisation.similarity.avg_similarity_layer_list

.. autofunction:: vae_ld.visualisation.similarity.plot_tsne

Visualise IDES
--------------

.. warning::
   These functions relies on the layer names of the current implementation and are not compatible with custom models.


.. autofunction:: vae_ld.visualisation.ides.plot_latents_ides

.. autofunction:: vae_ld.visualisation.ides.plot_data_ides

.. autofunction:: vae_ld.visualisation.ides.plot_layers_ides

Visualisation utils
-------------------

.. autofunction:: vae_ld.visualisation.utils.save_figure

.. autofunction:: vae_ld.visualisation.similarity.aggregate_similarity

.. autofunction:: vae_ld.visualisation.ides.aggregate_ides
