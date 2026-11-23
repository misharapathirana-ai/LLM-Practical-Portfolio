# Lab 01 - Sentiment Analysis Using Traditional Machine Learning

## Laboratory Session
Lab 01 - PUSL3205 Large Language Models and Generative AI

## Objective
Build a basic NLP pipeline from raw text to a trained classifier: preprocess text, create a Bag-of-Words representation, train Logistic Regression and a Linear SVM, and evaluate them on unseen sentences.

## Tasks and Implementation
- Dataset: UCI Sentiment Labelled Sentences (Amazon, IMDB, Yelp), https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences
- Preprocessing: lowercase, removed apostrophes and punctuation, collapsed extra whitespace, created a clean_text column
- Exploration: class balance, sentence length, positive and negative examples, most frequent words
- Split: 80/20, stratified, random_state=42
- Features: Bag-of-Words (also compared bigrams and TF-IDF)
- Models: Logistic Regression, Linear SVM
- Evaluation: Accuracy, Precision, Recall, F1, confusion matrix

Code: [notebook/Lab01_Sentiment_Analysis.ipynb](notebook/Lab01_Sentiment_Analysis.ipynb)

## Results and Evidence
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.796 | 0.803 | 0.782 | 0.793 |
| Linear SVM | 0.797 | 0.800 | 0.792 | 0.796 |

| Feature set (Logistic Regression) | Accuracy | F1 |
|---|---|---|
| BOW (unigrams) | 0.796 | 0.793 |
| BOW (unigrams + bigrams) | 0.806 | 0.805 |
| TF-IDF (unigrams) | 0.787 | 0.785 |
| TF-IDF (unigrams + bigrams) | 0.791 | 0.791 |

Screenshots are in the `screenshots/` folder.

## Observations

**Model comparison.** Logistic Regression achieved 79.6% accuracy and an F1 of 79.3%. The Linear SVM achieved 79.7% accuracy and an F1 of 79.6%. The difference is very small on this test set. The test set was balanced (299 negative, 298 positive) and results come from a single 80/20 split. No statistical significance test was performed.

**Feature comparison.** With Logistic Regression, unigrams plus bigrams performed best (80.6% accuracy, F1 80.5%). Bigrams may have helped capture short phrases such as "not good". TF-IDF did not help (78.7% and 79.1%).

**Positive and negative sentences.** Words only in the positive top list: best, excellent, great, love, well. Words only in the negative top list: bad, dont, even, time, would. TODO: add sentence-length result.

**Misclassified sentences.** The Linear SVM misclassified 121 of 597 test sentences (about 20.3%). Errors included negation, mixed opinions and context-dependent wording. TODO: add examples and checked explanations.

**New sentences.** "Oh great, another delay. Just perfect." got P(positive) = 0.97 despite being sarcastic. "I did not hate it" got 0.08, so the model reacted to "hate" and ignored "not". "The film was fine I suppose" got 0.84, and the dataset has no neutral class.

**Limitations.** Bag-of-Words uses word counts and ignores word order, so negation, sarcasm, idioms and mixed opinions cause errors. Transformer models use attention to represent tokens in context, which can help, but their performance would also need to be measured.

## Learning Outcomes
- Building a full NLP pipeline: cleaning, splitting, vectorising, training and evaluating
- Fitting the vectorizer on training data only to avoid leaking test information
- Using error analysis to learn more than a single accuracy number shows
- Why word-count models struggle with context, which motivates Transformers
