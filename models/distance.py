"""Classes of distance measure for model type A
"""
# Author: Yinchen Wu <yinchen@uchicago.edu>

import numpy as np

class Fourier:
    """ The function class for Fourier measure
    good for contextual anomolies
    ----------
    power: int, optional (default = 2)
        Lp norm for dissimiarlity measure considered
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    detector: Object classifier
        the anomaly detector that is used
    """
    def __init__(self, power = 2):
        self.decision_scores_  = []
        self.power = power
    def set_param(self):
        '''update the parameters with the detector that is used 
        since the FFT measure doens't need the attributes of detector
        or characteristics of X_train, the process is omitted. 
        '''

        return self

    def measure(self, X2, X3, start_index):
        """Obtain the SSA similarity score.
        Parameters
        ----------
        X2 : numpy array of shape (n, )
            the reference timeseries
        X3 : numpy array of shape (n, )
            the tested timeseries
        index: int, 
        current index for the subseqeuence that is being measured 
        Returns
        -------
        score: float, the higher the more dissimilar are the two curves 
        """       
 
        power = self.power
        X2 = np.array(X2)
        X3 = np.array(X3)
        if len(X2) == 0:
            score = 0
        else:
            X2 = np.fft.fft(X2)
            X3 = np.fft.fft(X3)
            score = np.linalg.norm(X2 - X3, ord = power)/len(X3)
        self.decision_scores_.append((start_index, score))
        return score
