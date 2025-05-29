# Image Classification & Transfer-Learning Pipeline

A self-contained end-to-end project demonstrating classic and deep learning approaches to image recognition.  
Implemented in Python using NumPy, scikit-learn, and TensorFlow/Keras, and developed in Google Colab.

---

## Overview

- **Task 1: Handwritten Digit Recognition (MNIST)**  
  - Custom loaders for IDX-format MNIST files  
  - Dimensionality reduction: PCA (50 components) & LDA (9 components)  
  - Kernel methods: SVM (linear, polynomial-3, RBF)  
  - Multiclass logistic regression (softmax)  
  - LeNet-style convolutional neural network (10 epochs; ≥99% test accuracy)

- **Task 2: Transfer Learning (10-Monkey-Species)**  
  - Baseline CNN trained from scratch  
  - Frozen VGG16 feature extractor + custom classifier head  
  - Fine-tuning top VGG16 layers with a low learning rate  
  - Early stopping to prevent overfitting
    
## Installation

1. **Clone this repo**  
   '''bash
   git clone https://github.com/ananyad5/face-me.git
   cd face-me
   pip install numpy scikit-learn tensorflow>=2.5'''
   

2. **Download Data**
MNIST: place train-images-idx3-ubyte(.gz) and train-labels-idx1-ubyte(.gz) (and their test counterparts) under data/mnist/.
10-Monkey-Species: download and extract into data/monkey/ with subfolders training/ and validation/.

3. **Open and run** all cells in the *.ipynb files
   
## References

1. LeCun et al., “Gradient‐based learning applied to document recognition”
   
