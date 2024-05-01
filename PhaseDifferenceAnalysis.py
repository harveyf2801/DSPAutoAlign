# Import the relevant modules
import numpy as np
from scipy.fft import fft, ifft

def phase_difference_analysis(ref, sig):
    """
    Perform phase difference analysis alignment of the input signal with respect to a reference signal.

    Parameters:
    - ref: Reference signal
    - sig: Input signal to be aligned

    Returns:
    - sig_aligned: Aligned signal
    """

    # Transform the signals to the frequency domain using FFT
    N = len(ref)
    FFT_ref = fft(ref, n=N)
    FFT_sig = fft(sig, n=N)

    # Compute the phase difference
    phase_diff = np.angle(FFT_sig) - np.angle(FFT_ref)

    # Compensate the phase difference for sig (using ref as a reference)
    FFT_sig_aligned = np.abs(FFT_sig) * np.exp(1j * (np.angle(FFT_sig) - phase_diff))

    # Transform back to the time domain using the inverse FFT
    sig_aligned = np.real(ifft(FFT_sig_aligned))

    return sig_aligned