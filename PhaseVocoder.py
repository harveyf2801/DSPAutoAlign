# Import the relevant modules
import librosa
import numpy as np

def phase_vocoder(ref, sig):
    """
    Apply phase vocoder technique to align the phase of a signal with a reference signal.

    Parameters:
    - ref: Reference signal with the desired phase
    - sig: Input signal to be aligned with the reference phase

    Returns:
    - aligned_source: Aligned signal with phase adjusted to match the reference signal
    """

    # Compute the Short-Time Fourier Transform (STFT) of both signals
    N = len(ref)

    stft_source = librosa.stft(sig)
    stft_reference = librosa.stft(ref)

    # Extract magnitudes and phases
    mag_source, phase_source = librosa.magphase(stft_source)
    mag_reference, phase_reference = librosa.magphase(stft_reference)

    # Calculate the phase difference
    phase_diff = phase_reference - phase_source

    # Apply the phase difference to the source signal
    stft_aligned_source = mag_source * np.exp(1j * (phase_source + phase_diff))

    # Inverse STFT to get the time-domain signal
    aligned_source = librosa.istft(stft_aligned_source, length=N)

    return aligned_source