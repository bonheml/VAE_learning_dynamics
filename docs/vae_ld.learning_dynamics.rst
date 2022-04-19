##########################
Learning dynamics
##########################

Analyse layer activations of deep learning models to monitor their learning dynamics.
One can also analyse the complexity of a dataset there.

Representational similarity
===========================

Linear Centered Kernel Alignment (CKA)
--------------------------------------
.. autoclass:: vae_ld.learning_dynamics.cka.CKA
   :members:
   :undoc-members:

Procrustes similarity
---------------------

.. automodule:: vae_ld.learning_dynamics.procrustes
   :members:
   :undoc-members:
   :show-inheritance:

Intrinsic dimension estimation (IDE)
====================================

Maximum likelihood estimation (MLE)
-----------------------------------
.. autoclass:: vae_ld.learning_dynamics.intrinsic_dimension_estimators.MLE
   :members:
   :undoc-members:

TwoNN
-----
.. autoclass:: vae_ld.learning_dynamics.intrinsic_dimension_estimators.TwoNN
   :members:
   :undoc-members:

Hidalgo
-------
.. autoclass:: vae_ld.learning_dynamics.intrinsic_dimension_estimators.Hidalgo
   :members:
   :undoc-members:

