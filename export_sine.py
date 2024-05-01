import numpy as np
from scipy.io import wavfile
from pathlib import Path

from main import SAMPLERATE, AUDIO_PATH, scale


if __name__ == "__main__":
    frequency = 440
    harm_freq = 260
    dur = 1
    t = np.linspace(0, dur, int(dur * SAMPLERATE), endpoint=False)
    ref = np.sin(2 * np.pi * frequency * t)
    sig = np.sin(2 * np.pi * frequency * t + (np.pi*0.9))

    sine_path1 = Path(AUDIO_PATH, f'sine_input.wav')
    wavfile.write(sine_path1, SAMPLERATE, scale(sig))

    sine_path2 = Path(AUDIO_PATH, f'sine_target.wav')
    wavfile.write(sine_path2, SAMPLERATE, scale(ref))