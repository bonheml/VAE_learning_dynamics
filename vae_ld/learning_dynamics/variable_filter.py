import numpy as np
import pandas as pd
import tensorflow as tf

from vae_ld.learning_dynamics import logger


def filter_variables(mean_vars, save_file=None, var_threshold=0.1, mean_error_range=0.1):
    """ Filter the latent variables of a VAE by type: active, mixed and passive

    Parameters
    ----------
    mean_vars: the activations of the mean layer
    save_file: the file where the results will be saved, if None, results are returned without saving
    var_threshold: the variance threshold below which a variable of the variance layer is considered low
    mean_error_range: the threshold below which a variable of the mean layer is considered low

    Returns
    -------
    pandas.Dataframe
        A dataframe containing the occurences of each variable type, their indexes, means and variance values.
    """
    z_vars = mean_vars.numpy().T
    scores = {}

    num_codes = z_vars.shape[0]
    variances = np.var(z_vars, axis=1)
    means = np.mean(z_vars, axis=1)
    assert num_codes == variances.shape[0] == means.shape[0]

    all_idxs = set(list(range(num_codes)))
    low_var_idxs = set(list(np.where(variances < var_threshold)[0]))
    higher_var_idxs = all_idxs - low_var_idxs
    # We only need to check the upper bound as variance will always be positive
    zero_mean_idxs = set(list(np.where(means <= mean_error_range)[0]))
    one_mean_idxs = set(list(np.where(((1 - mean_error_range) <= means) & (means <= (1 + mean_error_range)))[0]))
    mixed_means_idxs = all_idxs - (zero_mean_idxs | one_mean_idxs)

    scores["passive_variables_idx"] = list(low_var_idxs.intersection(one_mean_idxs))
    scores["active_variables_idx"] = list(low_var_idxs.intersection(zero_mean_idxs))
    scores["mixed_variables_idx"] = list(mixed_means_idxs | higher_var_idxs)
    scores["passive_variables"] = len(scores["passive_variables_idx"])
    scores["active_variables"] = len(scores["active_variables_idx"])
    scores["mixed_variables"] = len(scores["mixed_variables_idx"])
    scores["variances"] = variances.tolist()
    scores["means"] = means.tolist()

    logger.info("Found {} passive variables, {} mixed variables, and {} active variables".format(
        scores["passive_variables"], scores["mixed_variables"], scores["active_variables"]))

    checksum = scores["passive_variables"] + scores["active_variables"] + scores["mixed_variables"]
    try:
        assert checksum == num_codes
    except AssertionError as _:
        raise AssertionError("Total number of indexes is {} instead of {}".format(checksum, num_codes))

    scores = {k: [v] for k, v in scores.items()}
    df = pd.DataFrame(scores)
    if save_file is not None:
        df.to_csv(save_file, sep="\t", index=False)
    return df
