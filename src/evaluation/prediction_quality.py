import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import mutual_info_score

def entropy_metric(probabilities):
    """
    Calculate the entropy of the given probabilities.
    """
    return -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)

def log_loss_metric(probabilities, labels):
    """
    Calculate the log loss of the given probabilities and labels.
    """
    return log_loss(labels, probabilities)

def mutual_info_metric(probabilities, labels):
    """
    Calculate the mutual information of the given probabilities and labels.
    """
    return mutual_info_score(labels, probabilities)

def accuracy_metric(predictions, labels):
    """
    Calculate the accuracy of the given predictions and labels.
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    return np.mean(predictions == labels)



