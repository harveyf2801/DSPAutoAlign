import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dissertation_plot_style.mplstyle')

def plot_time_difference(ref, sig, sig_aligned, samplerate):
    ## Plotting the original and aligned signals

    ts = 1 / samplerate
    dur = len(ref)/samplerate
    t = np.arange(0, dur, ts)

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(t, ref)
    plt.title('Original Reference')

    plt.subplot(3, 1, 2)
    plt.plot(t, sig)
    plt.title('Original Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t, sig_aligned)
    plt.title('Phase-Aligned Signal')

    plt.tight_layout()
    plt.show()