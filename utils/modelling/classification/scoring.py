from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    brier_score_loss
)

# Dictionary mapping scoring names to sklearn functions
SCORER_FUNCTIONS = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score,
    'log_loss': log_loss,
    'matthews_corrcoef': matthews_corrcoef,
    'balanced_accuracy': balanced_accuracy_score,
    'cohen_kappa': cohen_kappa_score,
    'brier_score': brier_score_loss
}


def get_scorer(name):
    """
    Retrieve the scoring function corresponding to the given name.

    Parameters
    ----------
    name : str
        Name of the scoring metric (must be in SCORER_FUNCTIONS).

    Returns
    -------
    function
        Corresponding scoring function from sklearn.

    Raises
    ------
    ValueError
        If the metric name is not supported.
    """
    if name not in SCORER_FUNCTIONS:
        raise ValueError(f"Unsupported scoring method: '{name}'. "
                         f"Available options: {list(SCORER_FUNCTIONS.keys())}")
    return SCORER_FUNCTIONS[name]
