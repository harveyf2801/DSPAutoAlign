# Import the relevant modules
import numpy as np
from scipy.signal import correlate

def cross_correlation(ref, sig):
    """
    Perform cross-correlation on two signals to find time lag and apply phase/time alignment.

    Parameters:
    - ref: Reference signal
    - sig: Input signal to be aligned

    Returns:
    - aligned_sig: Aligned signal based on cross-correlation
    """

    # Calculate the cross-correlation
    cross_corr = correlate(sig, ref, mode='full')

    # Find the time lag with maximum cross-correlation
    time_lag = np.argmax(cross_corr) - (len(sig) - 1)

    # Apply phase/time alignment
    aligned_sig = np.roll(sig, -time_lag)

    return aligned_sig