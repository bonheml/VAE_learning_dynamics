import gudhi
from vae_ld.learning_dynamics import logger
from vae_ld.visualisation.utils import save_figure


def compute_persistence(X, save_path, dim_reductor=None, use_tex=False):
    """ Compute and plot the persistence barcodes of `X` using `Gudhi <https://gudhi.inria.fr/python/latest/>`_.

    Parameters
    ----------
    X : np.array
        A (n_example, n_features) matrix
    save_path : str
        The absolute path where the result will be saved
    dim_reductor : object or None, optional
         Dimensionality reductor (e.g., sklearn.decomposition.PCA) that will be used on `X` before computing the
         persistence barcodes if not None. Default None.
    use_tex : bool, optional
        If True, allow latex usage for plot generation. Default False.

    Returns
    -------
    None
    """
    gudhi.persistence_graphical_tools._gudhi_matplotlib_use_tex = use_tex
    if dim_reductor is not None:
        logger.info("reducing dimensionality")
        X = dim_reductor.fit_transform(X)
    logger.info("Creating Vietoris-Rips complex")
    split_path = save_path.split(".")
    for s in [1.5, 3.5, 3.5]:
        fname = "{}_sparsity_{}".format("".join(split_path[:-1]), s)
        rips = gudhi.RipsComplex(points=X, sparse=s)
        logger.info("Creating simplex tree")
        simplex_tree = rips.create_simplex_tree()
        logger.info("Computing persistence")
        diag = simplex_tree.persistence()
        logger.info("Betti numbers are {}".format(simplex_tree.betti_numbers()))
        gudhi.plot_persistence_barcode(diag, legend=True)
        save_figure("{}.pdf".format(fname))
