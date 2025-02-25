---
title: "Disaster Tweet Classification using BERT"
---

# Overview
This project implements a disaster tweet classification model using BERT (Bidirectional Encoder Representations from Transformers). The model determines whether a tweet refers to a real disaster or not. It is built using PyTorch and Hugging Face's `transformers` library.

# Dataset
The dataset consists of tweets labeled as either:
- **1**: The tweet is about a real disaster.
- **0**: The tweet is not related to a real disaster.

The dataset includes the following columns:
- `id`: Unique identifier for each tweet
- `text`: The actual tweet content
- `keyword`: Disaster-related keywords (if available)
- `location`: Location information (if available)
- `target`: Label indicating if the tweet is about a real disaster

# Project Structure
```r
├── disaster_tweet_classification.Rmd  # Main code in R Markdown format
├── bert_predictions.csv               # Output file containing predictions
├── README.md                          # Project documentation
└── requirements.txt                    # Dependencies for the project
```

# Installation
To set up the project, install the required dependencies:
```r
install.packages("torch")
install.packages("transformers")
install.packages("reticulate")
```

# Model Implementation
The model is built using `BERT-base-uncased` for sequence classification. The training process includes:
1. **Tokenizing** the dataset using the BERT tokenizer.
2. **Creating PyTorch Datasets and DataLoaders** for training, validation, and testing.
3. **Fine-tuning the BERT model** for classification.
4. **Evaluating the model** using accuracy and classification reports.
5. **Making predictions** on test data and new tweets.

# Training
- The dataset is split into training (90%) and validation (10%).
- The training loop optimizes the model using the AdamW optimizer.
- The model is trained for **3 epochs**.
- The best model is selected based on validation accuracy.

# Prediction
The model predicts whether a given tweet is about a real disaster. Example usage:
```r
sample_tweet <- "There's been a major earthquake in the city center. Many buildings have collapsed."
prediction <- predict_tweet(sample_tweet)
print(paste("Prediction:", ifelse(prediction == 1, "Real Disaster", "Not a Real Disaster")))
```

# Output
- Predictions are saved to `bert_predictions.csv`.
- The classification report provides accuracy, precision, recall, and F1-score.

# Future Improvements
- Experiment with different transformer models (e.g., `RoBERTa`, `DistilBERT`).
- Optimize hyperparameters for improved accuracy.
- Use data augmentation techniques to improve robustness.
- Deploy the model using a simple API.

# Acknowledgments
This project uses the `transformers` library by Hugging Face and PyTorch for deep learning.
