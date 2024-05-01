import pandas as pd
import torch
import numpy as np

from main import method_options
from dataset import TestAudioDataset
from losses import MultiResolutionSTFTLoss, MSELoss
from evaluation_metrics import db_rms, dB_peak, thdn, mse_loss

# Metrics
losses = {
    'MR-STFT': MultiResolutionSTFTLoss(),
    'MSE': mse_loss
}

loudness = {
    'RMS': db_rms,
    'Peak': dB_peak
}

quality = {
    'THDN': thdn
}

if __name__ == '__main__':
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Set path constants
    ANN = pd.read_csv('/home/hf1/Documents/soundfiles/annotations.csv')
    AUDIO_DIR = '/home/hf1/Documents/soundfiles/SDDS_segmented_Allfiles/'

    total_loss = {'MR-STFT': 0, 'MSE': 0}
    total_loudness = {'RMS': 0, 'Peak': 0}
    total_quality = {'THDN': 0}

    dataset = TestAudioDataset(ANN, AUDIO_DIR, 44100)
    dataset_length = len(dataset)

    for key, method in method_options.items():
        for i in range(dataset_length):
            input_sig, target_sig = dataset[i]
            input_sig = input_sig.squeeze(0).numpy()
            target_sig = target_sig.squeeze(0).numpy()
        
            filtered_sig = method(target_sig, input_sig)
            mix = (target_sig + filtered_sig) / 2

            for loss_key, loss in losses.items():
                total_loss[loss_key] += float(loss(target_sig, filtered_sig))
            for loudness_key, loudness_func in loudness.items():
                total_loudness[loudness_key] += loudness_func(mix)
            for quality_key, quality_func in quality.items():
                total_quality[quality_key] += quality_func(target_sig, filtered_sig)

        print('\nMethod:', key)
        for loss_key, loss in total_loss.items():
            print(loss_key, loss/dataset_length)
        for loudness_key, loudness_val in total_loudness.items():
            print(loudness_key, loudness_val/dataset_length)
        for quality_key, quality_val in total_quality.items():
            print(quality_key, quality_val/dataset_length)

        total_loss = {'MR-STFT': 0, 'MSE': 0}
        total_loudness = {'RMS': 0, 'Peak': 0}
        total_quality = {'THDN': 0}
