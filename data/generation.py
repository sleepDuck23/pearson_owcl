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
DISTRIBUTION_FAMILIES = ('Gaussian', 'Laplace', 'Exponential', 'Uniform')


def sample_family(label, size, rng, means, stds, family=None):
    """Draw fresh observations with the specified population mean and std.

    A family can be assigned independently of the true regime label. Omitting
    it retains the historical 1=Gaussian, 2=Laplace, 3=Exponential, 4=Uniform API.
    """
    if label not in (1, 2, 3, 4):
        raise ValueError('Regime label must be 1, 2, 3, or 4')
    family = DISTRIBUTION_FAMILIES[label - 1] if family is None else family
    mean, std = means[label - 1], stds[label - 1]
    if family == 'Gaussian':
        values = rng.normal(mean, std, size)
    elif family == 'Laplace':
        values = rng.laplace(mean, std / np.sqrt(2), size)
    elif family == 'Exponential':
        values = rng.exponential(std, size) - std + mean
    elif family == 'Uniform':
        values = rng.uniform(mean - std * np.sqrt(3), mean + std * np.sqrt(3), size)
    else:
        raise ValueError(f'Unknown distribution family: {family}')
    return values[:, None]


def validate_tier_moments(tier, means, stds):
    means, stds = np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
    if (means.shape != (4,) or stds.shape != (4,) or not np.isfinite(means).all()
            or not np.isfinite(stds).all() or np.any(stds <= 0)):
        raise ValueError('Provide four finite means and four finite positive standard deviations')
    if tier in ('T1', 'T3') and not np.all(stds == stds[0]):
        raise ValueError(f'{tier} requires the same population std for all regimes')
    if tier in ('T2', 'T3') and not np.all(means == means[0]):
        raise ValueError(f'{tier} requires the same population mean for all regimes')
    if tier == 'T1' and len(np.unique(means)) != 4:
        raise ValueError('T1 requires four distinct population means')
    if tier == 'T2' and len(np.unique(stds)) != 4:
        raise ValueError('T2 requires four distinct population standard deviations')
    return means, stds


def generate_moment_matched_stream(tier='T1', segment_length=700, seed=42,
                                   means=None, stds=None, *, randomize_families=True,
                                   return_metadata=False):
    """Keep regime moments fixed and assign four distinct families once per seed.

    The chronological episode sequence stays fixed. Returns retain the same
    family and population moments but draw fresh observations. Family assignment
    and sampling use only the seeded generator, before either model is run.
    """
    defaults = {
        'T1': ((0, 3, -3, 6), (1, 1, 1, 1)),
        'T2': ((0, 0, 0, 0), (1, 2, 0.5, 3)),
        'T3': ((0, 0, 0, 0), (1, 1, 1, 1)),
    }
    if tier not in defaults:
        raise ValueError("Tier must be 'T1', 'T2', or 'T3'")
    if not isinstance(segment_length, (int, np.integer)) or segment_length < 1:
        raise ValueError('Segment length must be a positive integer')
    default_means, default_stds = defaults[tier]
    means, stds = validate_tier_moments(
        tier, default_means if means is None else means,
        default_stds if stds is None else stds)
    rng = np.random.RandomState(seed)
    families = (rng.permutation(DISTRIBUTION_FAMILIES).tolist() if randomize_families
                else list(DISTRIBUTION_FAMILIES))
    data = np.concatenate([sample_family(label, segment_length, rng, means, stds,
                                         family=families[label - 1])
                           for label in SEQUENCE]).ravel()
    result = data, list(range(segment_length, len(data), segment_length)), list(SEQUENCE)
    if return_metadata:
        metadata = dict(tier=tier, seed=seed, segment_length=int(segment_length),
                        randomize_families=bool(randomize_families),
                        regimes=[dict(regime=i + 1, distribution=family,
                                      mean=float(means[i]), std=float(stds[i]))
                                 for i, family in enumerate(families)])
        return (*result, metadata)
    return result
