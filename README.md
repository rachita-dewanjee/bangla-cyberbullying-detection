# Cyberbullying Detection in Bengali Social Media Comments

Undergraduate thesis work by Rachita Dewanjee.

This project classifies Bengali (Bangla) online comments into types of cyberbullying using natural
language processing and machine learning.

> **Content warning:** the dataset and some notebook outputs (for example, the most frequent words per
> class) contain offensive and abusive language, because that is what the models learn to detect.

## Dataset
44,001 Bengali social-media comments (Bangla online comments dataset), each labelled with one of five classes:

| Class | Comments |
|---|---|
| not bully | 15,340 |
| troll | 10,462 |
| sexual | 8,928 |
| religious | 7,577 |
| threat | 1,694 |

The dataset is not included in this repository.

## What the notebook does
[`bangla_cyberbullying_detection.ipynb`](bangla_cyberbullying_detection.ipynb):
1. **Data analysis:** class distribution, documents, words and unique words per class, most frequent
   words, comment-length distribution
2. **Preprocessing (Bengali NLP):** punctuation removal, stop-word removal and word normalization with
   [`sbnltk`](https://pypi.org/project/sbnltk/), keeping only Bengali characters, and removing comments
   with fewer than two words (42,754 comments remain)
3. **Feature extraction:** TF-IDF with unigram, bigram (1–2) and trigram (1–3) features
4. **Models:** Logistic Regression, Decision Tree, Random Forest, Multinomial Naive Bayes, KNN and
   linear SVM, each tuned per n-gram setting, with an 80/20 train/test split and evaluation by
   accuracy, precision, recall and F1-score

## Tech
Python, pandas, NumPy, scikit-learn, sbnltk, Matplotlib, Seaborn, Google Colab.

## Running
The notebook was run on Google Colab and reads the dataset from Google Drive
(`/content/drive/MyDrive/Colab Notebooks/bangla_online_comments_dataset.csv`). Change that path to
where your copy of the dataset is. Run the cells from top to bottom.
