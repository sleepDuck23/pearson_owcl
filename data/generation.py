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


# Distribution families and recurring sequence used by the T1/T2/T3 experiments.
SEQUENCE = (1, 2, 3, 4, 2, 1, 4, 3, 1, 4, 2, 3)


def sample_family(label, size, rng, means, stds):
    mean, std = means[label - 1], stds[label - 1]
    if label == 1:
        values = rng.normal(mean, std, size)
    elif label == 2:
        values = rng.laplace(mean, std / np.sqrt(2), size)
    elif label == 3:
        values = rng.exponential(std, size) - std + mean
    elif label == 4:
        values = rng.uniform(mean - std * np.sqrt(3), mean + std * np.sqrt(3), size)
    else:
        raise ValueError('Family label must be 1, 2, 3, or 4')
    return values[:, None]


def generate_moment_matched_stream(tier='T1', segment_length=700, seed=42,
                                   means=None, stds=None):
    defaults = {
        'T1': ((0, 3, -3, 6), (1, 1, 1, 1)),
        'T2': ((0, 0, 0, 0), (1, 2, 0.5, 3)),
        'T3': ((0, 0, 0, 0), (1, 1, 1, 1)),
    }
    if tier not in defaults:
        raise ValueError("Tier must be 'T1', 'T2', or 'T3'")
    if segment_length < 1:
        raise ValueError('Segment length must be positive')
    default_means, default_stds = defaults[tier]
    means = default_means if means is None else means
    stds = default_stds if stds is None else stds
    rng = np.random.RandomState(seed)
    data = np.concatenate([sample_family(label, segment_length, rng, means, stds)
                           for label in SEQUENCE]).ravel()
    return data, list(range(segment_length, len(data), segment_length)), list(SEQUENCE)

