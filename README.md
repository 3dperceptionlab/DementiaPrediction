<h1 align="center">A Deep Learning-Based Multimodal Architecture to predict Signs of Dementia</h3>

<p align="center">
  A Deep Learning-Based Multimodal Architecture to predict Signs of Dementia
  <br>
  <a href="https://www.sciencedirect.com/science/article/pii/S0925231223005362"><img alt="Neurocomputing" src="https://img.shields.io/badge/Neurocomputing-2023-blue.svg"></a> 
</p>

This repository contains the official implementation of the paper *"A Deep Learning-Based Multimodal Architecture to Predict Signs of Dementia"* published in **Neurocomputing 2023**. The project explores a pipeline that leverages various approaches to predict dementia using multiple sources of information from the DementiaBank Pitt Corpus dataset.

## Overview

The **DementiaBank Pitt Corpus dataset**, provided by the TalkBank project, includes audio recordings and their corresponding transcriptions from English speakers. 

Key features of this implementation:
- **Mel spectrograms** are extracted from the audio recordings.
- **BERT embeddings** are derived from the text transcriptions.
- Several approaches are proposed and evaluated to develop a robust multimodal model combining audio and text modalities.

To access the DementiaBank Pitt Corpus dataset, you must contact the TalkBank project and request permission to use their data.

## Repository Structure

The project is organized as follows:
- `preprocess/`: Contains preprocessing scripts required to prepare the DementiaBank Pitt Corpus data for training.
- `modules/`: Includes training scripts for models focusing on audio, text, and multimodal modalities. Each script handles dataset preparation and model definition tailored to its specific modality.

## Requirements

The project utilizes the following pre-configured Docker image: [huggingface/transformers-pytorch-gpu](https://hub.docker.com/r/huggingface/transformers-pytorch-gpu). This image includes the necessary dependencies such as:
- `torchvision`
- `torchaudio`
- `transformers`
- `librosa`

## Usage

1. **Preprocessing**:  
   Run the preprocessing scripts in the `preprocess/` directory to process the DementiaBank Pitt Corpus dataset. This step prepares the data for use in the models.

2. **Training**:  
   Use the scripts in the `modules/` directory to train models for individual modalities (audio or text) or the multimodal architecture.

## Notes

- Ensure you have obtained the required permissions from the TalkBank project to use the DementiaBank Pitt Corpus dataset.
- For more details on the methods and results, please refer to the *Neurocomputing 2023* publication.


## Citation

```bibtex
@article{ORTIZPEREZ2023126413,
title = {A Deep Learning-Based Multimodal Architecture to predict Signs of Dementia},
journal = {Neurocomputing},
volume = {548},
pages = {126413},
year = {2023},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2023.126413},
url = {https://www.sciencedirect.com/science/article/pii/S0925231223005362},
author = {David Ortiz-Perez and Pablo Ruiz-Ponce and David Tomás and Jose Garcia-Rodriguez and M. Flores Vizcaya-Moreno and Marco Leo},
}
```