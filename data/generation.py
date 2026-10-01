import numpy as np

def generate_mean_shift_data(length, changepoints, means=None, std=1.0, seed=None):
    """
    Generates 1D time-series data with mean shifts at specified time indices.
    
    Args:
        length (int): Total number of data points to generate.
        changepoints (list of int): Exact time indices where the mean changes.
        means (list of float, optional): The target means for each segment. 
                                         Must have length == len(changepoints) + 1.
                                         If None, random means are generated.
        std (float): Standard deviation of the Gaussian noise.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: The generated 1D data array.
        list: The true changepoint locations.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Ensure changepoints are sorted and valid
    changepoints = sorted(changepoints)
    if changepoints and (changepoints[0] <= 0 or changepoints[-1] >= length):
        raise ValueError("Changepoints must be strictly greater than 0 and less than 'length'.")
        
    num_segments = len(changepoints) + 1
    
    # If no means are provided, generate a random walk of means to ensure distinct shifts
    if means is None:
        # Start at 0, shift by a random amount between [-3*std, 3*std] at each change
        shifts = np.random.uniform(-4 * std, 4 * std, size=num_segments)
        # Force the first mean to be 0 for a clean baseline
        shifts[0] = 0.0 
        means = np.cumsum(shifts)
    elif len(means) != num_segments:
        raise ValueError(f"Expected {num_segments} means, but got {len(means)}.")
        
    data = np.zeros(length)
    
    # Create segment boundaries: [0, cp1, cp2, ..., length]
    boundaries = [0] + changepoints + [length]
    
    # Generate data segment by segment
    for i in range(num_segments):
        start = boundaries[i]
        end = boundaries[i+1]
        size = end - start
        
        data[start:end] = np.random.normal(loc=means[i], scale=std, size=size)
        
    return data, changepoints



def generate_regime_stream(length, changepoints, means, stds, seed=None):
    """
    Generates 1D time-series data with specific means and standard deviations for each regime.
    
    Args:
        length (int): Total number of data points to generate.
        changepoints (list of int): Exact time indices where the regime changes.
        means (list of float): The target means for each segment. 
                               Must have length == len(changepoints) + 1.
        stds (list of float): The target standard deviations for each segment.
                              Must have length == len(changepoints) + 1.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: The generated 1D data array.
        list: The true changepoint locations.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Ensure changepoints are sorted and physically valid
    changepoints = sorted(changepoints)
    if changepoints and (changepoints[0] <= 0 or changepoints[-1] >= length):
        raise ValueError("Changepoints must be strictly greater than 0 and less than 'length'.")
        
    num_segments = len(changepoints) + 1
    
    # Validate that means and stds match the number of segments
    if len(means) != num_segments:
        raise ValueError(f"Expected {num_segments} means, but got {len(means)}.")
    if len(stds) != num_segments:
        raise ValueError(f"Expected {num_segments} stds, but got {len(stds)}.")
        
    data = np.zeros(length)
    
    # Create segment boundaries: [0, cp1, cp2, ..., length]
    boundaries = [0] + changepoints + [length]
    
    # Generate data segment by segment
    for i in range(num_segments):
        start = boundaries[i]
        end = boundaries[i+1]
        size = end - start
        
        # Sample from a Gaussian distribution with the specified mean and std
        data[start:end] = np.random.normal(loc=means[i], scale=stds[i], size=size)
        
    return data, changepoints


