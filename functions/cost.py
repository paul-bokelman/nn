import numpy as np

class CostFunction:
    def __init__(self, y_actual: np.ndarray, y_pred: np.ndarray):
        self.y_actual = y_actual
        self.y_pred = y_pred
    def mean_absolute_error(self):
        return mean_absolute_error(self.y_actual, self.y_pred)
    def mean_squared_error(self):
        return mean_squared_error(self.y_actual, self.y_pred)
    def root_mean_squared_error(self):
        return root_mean_squared_error(self.y_actual, self.y_pred)
    def root_mean_squared_log_error(self):
        return root_mean_squared_log_error(self.y_actual, self.y_pred)

# ---------------------------- MEAN ABSOLUTE ERROR ---------------------------- #
# - less sensitive to outliers
# - annoying for optimization because it's not differentiable

def mean_absolute_error(y_actual: np.ndarray, y_pred: np.ndarray):
    abs_error = np.abs(y_actual - y_pred)
    mean_abs_error = np.mean(abs_error)
    return mean_abs_error

# ---------------------------- MEAN SQUARED ERROR ---------------------------- #
# - sensitive to outliers, 
# - penalizes larger errors 
# - more useful when larger errors are more significant

def mean_squared_error(y_actual: np.ndarray, y_pred: np.ndarray):
    squared_error = (y_actual - y_pred) ** 2
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error

# ---------------------------- ROOT MEAN SQUARED ERROR ---------------------------- #
# - sensitive to outliers
# - not as sensitive as mean squared error

def root_mean_squared_error(y_actual: np.ndarray, y_pred: np.ndarray):
    squared_loss = (y_actual - y_pred) ** 2
    mean_squared_error = np.mean(squared_loss)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error

# ---------------------------- ROOT MEAN SQUARED ERROR ---------------------------- #
# - sensitive to outliers
# - not as sensitive as root mean squared error

def root_mean_squared_log_error(y_actual: np.ndarray, y_pred: np.ndarray):
    squared_log_loss = (np.log(y_actual + np.ones(y_actual.shape)) - np.log(y_pred + np.ones(y_pred.shape))) ** 2
    mean_squared_log_error = np.mean(squared_log_loss)
    root_mean_squared_log_error = np.sqrt(mean_squared_log_error)
    return root_mean_squared_log_error

