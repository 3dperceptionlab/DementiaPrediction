import warnings
import numpy as np
import librosa
import torch
import torchaudio
import torchvision
from PIL import Image
warnings.filterwarnings('ignore')

pittPath = '/Pitt/'

def extract_spectrogram(path):
  waveform, sample_rate = librosa.load(path)
  window_sizes = [25, 50, 100]
  hop_sizes = [10, 25, 50]

  specs = []

  for i in range(len(window_sizes)):
    window_length = int(round(window_sizes[i]*sample_rate/1000))
    hop_length = int(round(hop_sizes[i]*sample_rate/1000))

    clip = torch.Tensor(waveform)
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2205, win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
    eps = 1e-6
    spec = spec.numpy()
    spec = np.log(spec+ eps)
    spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
    specs.append(spec)

  specs = np.array(specs)
  np.save(path[:-4] + '.npy', specs)
  #specs = np.array(specs).reshape(-1, 128, 250)

def preprocessAudio():
  categories = ['controlCha.txt', 'dementiaCha.txt']
  for cat in categories:
    path = pittPath + cat

    index = open(path, 'r')
    files = index.readlines()
    for file in files:
      extract_spectrogram(file[:-4] + 'mp3')

if __name__ == "__main__":
    preprocessAudio()
