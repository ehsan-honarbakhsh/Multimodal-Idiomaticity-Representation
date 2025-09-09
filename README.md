# Multimodal Idiomatic Language Understanding

## Overview

This project develops a multimodal algorithm to address the challenges of understanding idiomatic language by integrating textual and visual information. It combines BERT for text encoding, EfficientNetB0 for image encoding, and zero-shot classification to predict whether a sentence (potentially containing idioms) corresponds to a specific image-caption pair. The algorithm uses a dataset of sentences, images, and captions, with labels indicating correct matches. Data augmentation with T5 paraphrasing enhances the training set, and an ensemble approach improves prediction accuracy.

The implementation is designed for platforms like Google Colab, leveraging TensorFlow, Hugging Face Transformers, and Sentence Transformers for processing. It evaluates performance using F1 score, precision, recall, and accuracy, and provides visualizations of predicted versus true image-sentence matches.

Key features:

- Multimodal model combining BERT (text) and EfficientNetB0 (image) embeddings.
- Data augmentation using T5 for paraphrasing sentences and captions.
- Zero-shot classification with BART for ensemble predictions.
- Custom Keras layer for BERT integration.
- Comprehensive evaluation and visualization of results.

## Requirements

- Python 3.8+
- Libraries:
  - TensorFlow (`tensorflow>=2.10`)
  - Transformers (`transformers`)
  - Sentence Transformers (`sentence-transformers`)
  - Pandas (`pandas`)
  - NumPy (`numpy`)
  - Matplotlib (`matplotlib`)
  - Pillow (`pillow`)
  - Scikit-learn (`scikit-learn`)

Install dependencies using:

```
pip install tensorflow transformers sentence-transformers pandas numpy matplotlib pillow scikit-learn
```

## Dataset

The dataset consists of CSV files (`train.csv`, `val.csv`, `test.csv`) and corresponding images, structured as follows:

- **CSV Files**: Located in `/content/drive/MyDrive/{train,val,test}/`
  - Columns: `sentence`, `image_name`, `image_caption`, `label` (1 for correct sentence-image match, 0 otherwise).
- **Images**: Stored in `/content/drive/MyDrive/{train,val,test}/images/`
- **Structure**: Each sentence has exactly three rows, with one correct label (`label=1`) and two incorrect (`label=0`).

Ensure the dataset is accessible (e.g., mounted on Google Drive for Colab) or adjust paths accordingly.

## Usage

1. Place the dataset in the specified directories (e.g., `/content/drive/MyDrive/train`, `/content/drive/MyDrive/val`, `/content/drive/MyDrive/test`).
   Note: If using Google Colab, ensure Google Drive is mounted.
2. The script will:
   - Load and clean the dataset, ensuring no duplicates and exactly one correct label per sentence.
   - Visualize a sample of test images with their compounds and labels.
   - Augment the training data using T5 paraphrasing (sentences and captions).
   - Preprocess text (BERT tokenizer) and images (EfficientNetB0 preprocessing).
   - Train a multimodal model combining text and image embeddings.
   - Perform zero-shot classification using BART.
   - Ensemble model predictions with zero-shot probabilities (0.7:0.3 weighting).
   - Evaluate performance with F1 score, precision, recall, and accuracy.
   - Visualize predicted versus true image-sentence matches for up to 5 samples.

Customization:

- Adjust `num_augmentations` (default: 2) and `similarity_threshold` (default: 0.8) for data augmentation.
- Modify `max_len` (default: 50) for BERT tokenization.
- Change `projection_dim` (default: 128) for embedding projection.
- Tune ensemble weights (default: 0.7 model + 0.3 zero-shot).
- Adjust training parameters: `epochs` (default: 10), `batch_size` (default: 32), or callbacks.

## Algorithm Details

### 1. Data Preprocessing

- **Data Cleaning**:
  - Remove duplicates based on `sentence` and `image_name`.
  - Ensure each sentence has exactly one correct label (`label=1`) and three rows.
- **Data Augmentation**:
  - Use T5 (`t5-base`) to paraphrase sentences and captions.
  - Filter paraphrases with cosine similarity ≥ 0.8 (using `all-MiniLM-L6-v2`).
  - Append valid paraphrases to the training dataset.
- **Text Processing**:
  - Tokenize sentences and captions using BERT tokenizer (`bert-base-uncased`, max length 50).
  - Return `input_ids` and `attention_mask` for model input.
- **Image Processing**:
  - Load images, resize to 224x224, and preprocess using EfficientNetB0's input normalization.

### 2. Multimodal Model

- **Inputs**:
  - Context: Sentence (`input_ids`, `attention_mask`).
  - Caption: Image caption (`input_ids`, `attention_mask`).
  - Image: Preprocessed image (224x224x3).
- **Text Encoder**:
  - Custom `BertLayer` wraps `TFBertModel` (`bert-base-uncased`) for Keras compatibility.
  - Apply global average pooling to BERT outputs for text embeddings.
- **Image Encoder**:
  - Use EfficientNetB0 (pretrained on ImageNet, no top) to extract image features.
  - Apply global average pooling and dense layer (128 units, ReLU) for image embedding.
- **Fusion**:
  - Concatenate image and caption embeddings.
  - Project context and image-caption embeddings to a common dimension (128 units, ReLU).
  - Concatenate all embeddings and pass through a dense layer (sigmoid) for binary classification.
- **Training**:
  - Optimizer: Adam.
  - Loss: Binary crossentropy.
  - Metrics: Accuracy.
  - Callbacks: Learning rate scheduler (exponential decay after 5 epochs), early stopping (patience=3).

### 3. Zero-Shot Classification

- Use BART (`facebook/bart-large-mnli`) for zero-shot classification.
- Predict probability that a sentence matches its caption versus "irrelevant."
- Combine with model predictions (0.7 model + 0.3 zero-shot) for ensemble.

### 4. Evaluation

- **Metrics**:

  - F1 score, precision, recall, accuracy (macro-averaged for balanced evaluation).

- **Prediction**:
  - For each sentence, select the image with the highest ensemble probability.
  - Compare predicted image names with true image names (where `label=1`).

### 5. Visualization

- **Dataset Visualization**:
  - Display test images with their compounds and labels in a grid.
- **Prediction Visualization**:
  - Show up to 5 samples of predicted versus true image-sentence pairs.
  - Include sentence and caption text below each image.

## Results

The script outputs:

- **Visualizations**:
  - Grid of test images with compounds and labels.
  - Predicted versus true image-sentence pairs for up to 5 samples.
- **Metrics**:
  - F1 score, precision, recall, and accuracy on the test set.
  - Performance depends on dataset quality and model training.

The model aims to accurately identify correct image-sentence pairs, leveraging multimodal information to handle idiomatic language.

## Limitations

- **Dataset Dependency**: Tailored for a specific dataset structure; modify paths for other datasets.
- **Computational Resources**: Requires GPU for efficient training and inference (BERT, EfficientNet, T5).
- **Augmentation Quality**: Paraphrasing depends on T5 and similarity threshold; low-quality paraphrases may degrade performance.
- **Evaluation**: Assumes exactly one correct image per sentence; adapt for other label distributions.

## The full description of the algorithm can be found in this

[Full Description can be found here](https://github.com/ehsan-honarbakhsh/Multimodal-Idiomaticity-Representation/blob/main/Docs/Multimodal%20Idiomaticity%20Representation%20.pdf)
