import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

def calculate_metrics(true_values, predicted_values):
    """
    计算各种评估指标
    
    Args:
        true_values: 真实值列表
        predicted_values: 预测值列表
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    spearman_corr, _ = spearmanr(true_values, predicted_values)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman_corr
    }
    
    return metrics

def print_metrics(metrics):
    """
    打印评估指标
    
    Args:
        metrics: 包含评估指标的字典
    """
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"R-squared: {metrics['r2']:.4f}")
    print(f"Spearman Correlation: {metrics['spearman']:.4f}")