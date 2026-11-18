# Variational Autoencoders on Medical Images

Convolutional VAE and β-VAE trained on 64×64 grayscale images from MedNIST (6 classes: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT). Lab project at Institut d'Optique Graduate School, done in a team of four.

## What it does

- Trains a convolutional VAE (PyTorch, 64-dimensional latent space)
- Compares β = 1, 4 and 10 (reconstruction vs. regularisation trade-off)
- Visualises the latent space with PCA and t-SNE, and interpolates between images
- Estimates the log-likelihood per class
- Generates new images by sampling from N(0, I)
- Analyses the best and worst reconstructions and inactive latent dimensions (posterior collapse)

## Results

- Test MSE: **0.0066**
- t-SNE shows clear clusters by image class
- Higher β gives a more compact latent space, with blurrier reconstructions

## Run it

The script was exported from a Google Colab notebook. The easiest way is to open it in Colab (GPU recommended).

```bash
pip install monai torch torchvision matplotlib scikit-learn tqdm
```

Then run `vae_mednist.py` (the dataset is downloaded automatically).

## Files

- `vae_mednist.py`: full code (data, model, training, analyses)
- `VAE.pdf`: full report (in French)

## Authors

Robin Jegard, Elias Veyrac, Alexis Demilly, Jianfei Fang

## Stack

Python · PyTorch · MONAI · scikit-learn · Matplotlib
