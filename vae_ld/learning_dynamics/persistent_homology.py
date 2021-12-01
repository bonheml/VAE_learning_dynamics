import gudhi
from vae_ld.learning_dynamics import logger
from vae_ld.learning_dynamics.utils import save_figure


def compute_persistence(X, save_path, dim_reductor=None):
    if dim_reductor is not None:
        logger.info("reducing dimensionality")
        X = dim_reductor.fit_transform(X)
    logger.info("Creating Vietoris-Rips complex")
    rips = gudhi.RipsComplex(points=X)
    logger.info("Creating simplex tree")
    simplex_tree = rips.create_simplex_tree()
    logger.info("Computing persistence")
    diag = simplex_tree.persistence()

    gudhi.plot_persistence_barcode(diag)

    save_figure(save_path)
