import tensorflow_probability as tfp
import tensorflow as tf
from scipy.optimize import linear_sum_assignment


def mcc(z_true, z_pred):
    """ Implementation of the Mean correlation coefficient as described in [1].
    Based on Khemakhem et al. `implementation <https://github.com/siamakz/iVAE/blob/master/lib/metrics.py>`_.

    Parameters
    ----------
    z_true
        The true generative factors
    z_pred
        The inferred latent variables

    Returns
    -------
    float
        A MCC score between 0 and 1.

    Examples
    --------
    >>> z_true = tf.random.normal(shape=(10000, 3))
    >>> z_pred = tf.random.normal(shape=(10000, 3))
    >>> mcc(z_true, z_pred)
        0.008850733
    >>> z_pred = z_true * 2 + 1
    >>> mcc(z_true, z_pred)
        1.0

    References
    ----------
    .. [1] Khemakhem, I., Kingma, D., Monti, R., & Hyvarinen, A. (2020, June). Variational autoencoders
           and nonlinear ica: A unifying framework. In International Conference on Artificial Intelligence
           and Statistics (pp. 2207-2217). PMLR.
    """
    corr = tfp.stats.correlation(z_true, z_pred)
    neg_abs_corr = -tf.abs(corr).numpy()
    rows, cols = linear_sum_assignment(neg_abs_corr)
    return -neg_abs_corr[rows, cols].mean()
