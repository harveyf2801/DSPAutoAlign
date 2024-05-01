# Import the relevant modules
import numpy as np
from scipy.signal import csd
from scipy.fft import fft, ifft

def cross_spectrum(ref, sig):
    """
    Perform phase alignment with cross-spectrum analysis.

    Parameters:
    - ref: Reference signal
    - sig: Signal to be aligned with the reference signal

    Returns:
    - aligned_sig: Aligned signal
    """
    
    # Take FFT of full signals
    N = len(ref)
    FFT_ref = fft(ref, n=N)
    FFT_sig = fft(sig, n=N)

    # Cross spectrum
    cross_spec = FFT_ref * np.conj(FFT_sig)

    # Extract phase difference
    phase_diff = np.angle(cross_spec)

    # Apply inverse shift
    FFT_sig_aligned = FFT_sig * np.exp(-1j * phase_diff)

    # Reconstruct time domain
    aligned_sig = np.real(ifft(FFT_sig_aligned))

    return aligned_sig
