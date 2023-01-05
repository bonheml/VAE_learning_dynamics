import numpy as np
from vae_ld.learning_dynamics import logger


class EffectiveRank:
    def __init__(self, erank=True):
        self.erank = erank

    def fit_transform(self, X):
        """Computes the effective rank of a representation using implementation from
        Implicit Regularization in Deep Learning May Not Be Explainable by Norms (Razin and Cohen, 2020).
        https://github.com/noamrazin/imp_reg_dl_not_norms

        Parameters
        ----------
        X : np.array
            A (n_example, n_features) matrix

        Returns
        -------
        float
            The effective rank of X
        """
        cov = np.cov(X, rowvar=False)
        logger.debug("Cov(X) = {}".format(cov))
        sv = np.linalg.svd(cov, compute_uv=False)
        logger.debug("Singular values : {}".format(sv))
        non_zero_sv = sv[sv != 0]
        rank = non_zero_sv.shape[0]
        logger.info("Rank: {}".format(rank))
        non_zero_sv_norm = non_zero_sv / non_zero_sv.sum()
        sv_entropy = -(non_zero_sv_norm * np.log(non_zero_sv_norm)).sum()
        erank = np.exp(sv_entropy)
        logger.info("Effective rank: {}".format(erank))
        return erank if self.erank else rank
