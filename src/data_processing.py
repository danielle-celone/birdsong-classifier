import librosa
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from pathlib import Path
import numpy as np
#load audio files and extract Mel-spectrograms

def plot_spectrogram_and_save(signal, sample_rate, output_path: Path):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    plt.figure(figsize=(10,4))
    img = librosa.display.specshow(spectrogram_db, y_axis='log', x_axis='time', sr=sample_rate, cmap='inferno')
    plt.colorbar(img, format='%+2.f dBFS')
    #plt.show()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)

def main():
    signal, sample_rate = librosa.load(Path('../data/raw/A-M') / 'aldfly/XC16964.mp3', sr=None)
    print("Sample rate: " + str(sample_rate))
    plot_spectrogram_and_save(signal, sample_rate, Path('../data/processed') / 'aldfly/XC16964.png')

if __name__ == '__main__':
    main()