import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa
from scipy.io import wavfile
from losses import MultiResolutionSTFTLoss, MSELoss

from evaluation_metrics import thdn, db_rms, dB_peak, mse_loss
from PlotPhaseDifference import plot_time_difference

# Audio options for alignment comparison
audio_options = [
    'sine',
    'snare1',
    'snare2',
    'snare3'
]

# Options for various alignment methods
method_options = [
    'cross_correlation',
    'phase_vocoder',
    'phase_difference_analysis',
    'cross_spectrum'
]

# Select the audio and alignment method to use
method = method_options[3]
audio = audio_options[2]

# Setting the audio constants
# * DURATION of None means the whole audio file will be used
SAMPLERATE = 44100
DURATION = None

LOSS = MultiResolutionSTFTLoss()

PLOTTING = False
PLAY_AUDIO = False
EVALUATION_METRICS = True
OUTPUT_AUDIO = False

# Creating the audio / loading in the audio files
if (audio == audio_options[0]):
    frequency = 440
    harm_freq = 260
    dur = 1
    t = np.linspace(0, dur, int(dur * SAMPLERATE), endpoint=False)
    ref = np.sin(2 * np.pi * frequency * t)
    sig = np.sin(2 * np.pi * frequency * t + (np.pi*0.9))

elif (audio == audio_options[1]):
    ref, _ = librosa.load(Path('C:/Users/hfret/Downloads/SDDS/Gretsch_BigFatSnare_AKG_414_BTM_Segment_10_peak_0.066.wav'), duration=DURATION, mono=True, sr=SAMPLERATE)
    sig, _ = librosa.load(Path('C:/Users/hfret/Downloads/SDDS/Gretsch_BigFatSnare_AKG_414_BTM_Segment_2_peak_0.051.wav'), duration=DURATION, mono=True, sr=SAMPLERATE)

elif (audio == audio_options[2]):
    ref, _ = librosa.load(Path('C:/Users/hfret/Downloads/SDDS/wooden15_BigFatSnare_Sennheiser_MD421_TP_Segment_7_peak_0.029.wav'), duration=DURATION, mono=True, sr=SAMPLERATE)
    sig, _ = librosa.load(Path('C:/Users/hfret/Downloads/SDDS/wooden15_BigFatSnare_Shure_SM57_BTM_Segment_7_peak_0.035.wav'), duration=DURATION, mono=True, sr=SAMPLERATE)

elif (audio == audio_options[3]):
    ref, _ = librosa.load(Path('C:/Users/hfret/Downloads/SDDS/YamahaMaple_BigFatSnare_Sennheiser_e614_TP_Segment_80_peak_0.163.wav'), duration=DURATION, mono=True, sr=SAMPLERATE)
    sig, _ = librosa.load(Path('C:/Users/hfret/Downloads/SDDS/YamahaMaple_BigFatSnare_AKG_414_BTM_Segment_80_peak_0.132.wav'), duration=DURATION, mono=True, sr=SAMPLERATE)

# Normalize the signals
ref /= np.max(np.abs(ref))
sig /= np.max(np.abs(sig))

# Check the polarity of the signals and invert if necessary
invert_pol = abs(float(LOSS(sig, ref))) > abs(float(LOSS(-sig, ref)))
if invert_pol: sig = -sig
print("The polarity of the signal", "needs" if invert_pol else "does not need", "to be inverted")

# Applying the selected alignment method
if (method == method_options[0]):
    from CrossCorrelation import cross_correlation
    filtered_sig = cross_correlation(ref.copy(), sig.copy())

elif (method == method_options[1]):
    from PhaseVocoder import phase_vocoder
    filtered_sig = phase_vocoder(ref.copy(), sig.copy())

elif (method == method_options[2]):
    from PhaseDifferenceAnalysis import phase_difference_analysis
    filtered_sig = phase_difference_analysis(ref.copy(), sig.copy())

elif (method == method_options[3]):
    from CrossSpectrum import cross_spectrum
    filtered_sig = cross_spectrum(ref.copy(), sig.copy())

# Creating mixes for the signals to listen to
mix1 = (sig + ref) / 2
mix2 = (filtered_sig + ref) / 2

if PLOTTING:
    # Plotting the original and aligned signals
    plot_time_difference(ref, sig, filtered_sig, SAMPLERATE)

if PLAY_AUDIO:
    sd.play(mix1, SAMPLERATE)
    sd.wait()

    time.sleep(0.5)

    sd.play(mix2, SAMPLERATE)
    sd.wait()

if EVALUATION_METRICS:
    losses = {
        'MR-STFT': LOSS,
        'MSE': mse_loss
    }

    loudness = {
        'RMS': db_rms,
        'Peak': dB_peak
    }

    quality = {
        'THD': thdn
    }

    print()

    # Calculating the loss before and after alignment
    for key, value in losses.items():
        print(f"{key} Loss: {float(value(sig, ref))} -> {float(value(filtered_sig, ref))}")

    print()
    
    # Calculating the loudness before and after alignment
    for key, value in loudness.items():
        print(f"{key} (dB) Loudness: {value(mix1)} -> {value(mix2)}")

    print()

    # Calculating the quality before and after alignment
    for key, value in quality.items():
        print(f"{key} Quality: {value(sig)}% -> {value(filtered_sig)}%")

# Exporting the audio files
if OUTPUT_AUDIO:
    traditional_path = Path(f'TraditionalDSPMethods', 'audio')

    def scale(arr):
        ''' Scale the audio signal to the range of a 16-bit integer '''
        return np.int16(arr * 32767)

    def bitcrush(arr, bits=16):
        ''' Quantize the audio signal to a lower bit depth '''
        # Calculate the maximum value based on the bit depth
        max_value = 2 ** bits

        # Quantize the audio signal based on the maximum value
        quantized_signal = np.floor(arr * max_value) / max_value
        return quantized_signal

    # Exports for MUSHRA testing
    # Export mix1 as a wav file reference
    mix1_path = Path(traditional_path, f'ref_{audio}.wav')
    wavfile.write(mix1_path, SAMPLERATE, scale(mix1))

    # Export mix1 as a wav file
    mix1_anchor1_path = Path(traditional_path, f'6bit_{audio}.wav')
    wavfile.write(mix1_anchor1_path, SAMPLERATE, scale(bitcrush(mix1, 6)))

    # Export mix1 as a wav file
    mix1_anchor2_path = Path(traditional_path, f'4bit_{audio}.wav')
    wavfile.write(mix1_anchor2_path, SAMPLERATE, scale(bitcrush(mix1, 4)))

    # Export mix2 as a wav file
    mix2_path = Path(traditional_path, f'{method}_{audio}.wav')
    wavfile.write(mix2_path, SAMPLERATE, scale(mix2))
