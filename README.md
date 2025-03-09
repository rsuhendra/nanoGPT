# nanoGPT

A minimal implementation of the Transformer architecture to build a language model that generates infinite Shakespeare-like text. This project is inspired by Andrej Karpathy's video, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).

---

## Overview

This project was designed to help me understand the Transformer architecture by building a language model from scratch. The model is trained on the complete works of Shakespeare and can generate text in a similar style.

---

## Files

### 1. `model.py`

Contains the Transformer architecture built from the ground up.

### 2. `train.py`

Trains the GPT model and saves the trained model as `model.pth`.

### 3. `run.py`

Loads the saved model (`model.pth`) and generates infinite Shakespeare-like text.

### 4. `colab.ipynb`

A Jupyter Notebook for running the project on Google Colab. If you don't have a GPU, I would highly advise running it on the Colab GPUs. 

### 5. `sample.txt`

Contains a sample of generated text. You can view this file to see the output without training the model.

---

## Dataset

The input dataset consists of the combined works of Shakespeare. The model is trained on this dataset to learn the patterns and structure of Shakespearean text.

---

## Usage

### 1. Training the Model

To train the model, run:
```bash
python train.py