# Multimodal Idiomaticity Representation
 Multimodal Idiomaticity Representation

In this project, a system needs to be developed to address the challenges associated with idiomatic language. The primary challenge is that some phrases can have multiple meanings, one that aligns with the literal interpretation of the words (e.g., a real rotten apple) and another that conveys a figurative meaning.
To tackle this, the algorithm's input will consist of a sentence containing a phrase that can be interpreted either literally or figuratively, along with three images and their corresponding captions. The algorithm should accurately understand the context and predict which image best represents the intended meaning of the phrase in the given sentence
Architecture Overview
The model leverages pre-trained models,BERT for text processing and EfficientNetB0 for image processing , and using transfer learning and fine-tuning for the image-sentence matching challenge.
Image Preprocessing
 The function loads an image from a file and preprocesses it to make it compatible with the EfficientNetB0 model, which is used later in the algorithm for image feature extraction
Clean DataFrame Function
This function cleans the input DataFrame to ensure data consistency and quality, which is essential for training and evaluating the model
Text Encoding (BERT):
Sentences and captions are tokenized, producing input_ids and attention_mask  because BERT requires tokenized inputs with attention masks to process text efficiently and focus only on meaningful tokens, ignoring padding.
A custom BertLayer built on TFBertModel generates sequence embeddings, which are reduced to 768-dimensional vectors via global average pooling.
Output: Embeddings for sentences and captions .
Image Encoding (EfficientNetB0):
Images preprocessed to 224x224x3 resolution.
Preprocessing images to match EfficientNetB0’s input requirements enables effective feature extraction, aligning with the model’s pre-training on ImageNet.
EfficientNetB0 extracts features, followed by global average pooling to produce a 1280-dimensional vector, which is then projected to 128 dimensions using a dense layer with ReLU activation.
A 128-dimensional image embedding.

Data Augmentation:
LLM-based Augmentation:The goal of this section is to augment the training dataset  by generating paraphrased versions of the 'sentence' and 'image_caption' columns.
This method uses an LLM (T5) to generate paraphrases and a sentence embedding model (Sentence-BERT) to filter them, ensuring they remain semantically close to the originals.
Zero-shot Prediction:
A pre-trained BART model performs zero-shot classification to score caption-sentence relevance, serving as an ensemble component.
Produce a set of probability scores (zero_shot_probs) that can be combined with fine-tuned model predictions.
BART-large-MNLI excels at zero-shot tasks because it can generalize entailment relationships to new, unseen data using a hypothesis template.
Multimodal Fusion:
Combination: The image and caption embeddings are concatenated and projected to 128 dimensions. The sentence embedding is similarly projected to 128 dimensions.
Final Combination: These two 128-dimensional vectors are concatenated into a 256-dimensional vector.
Prediction: A dense layer with sigmoid activation outputs a probability (0 to 1) indicating the likelihood of an image-sentence match.
Prediction and Evaluation:
The fine-tuned model predicts probabilities for test data.
Ensemble: Combines model predictions (weighted 0.7) with zero-shot predictions (weighted 0.3).
Evaluation: Measures performance using accuracy , F1 score,Precision and Recall ,supplemented by visualizations of predicted versus true image-sentence pairs.
