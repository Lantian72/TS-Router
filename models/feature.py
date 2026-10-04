from numpy.lib.stride_tricks import sliding_window_view

class Window:
    """ The class for rolling window feature mapping.
    Converts the original time series X into a matrix of sliding windows.
    """
    def __init__(self, window=100, stride=1):
        self.window = window
        self.stride = stride

    def convert(self, X):
        shape = (X.shape[0] - (self.window - 1), -1)
        windows = sliding_window_view(X, window_shape=self.window, axis=0).reshape(shape)[::self.stride, :]        
        return windows
