# Financial News Sentiment Analysis

A deep learning NLP project that predicts whether financial news headlines are **Positive** or **Negative** using **BiLSTM + Word2Vec embeddings**.

---

## 🚀 Project Overview

This project performs sentiment analysis on financial news headlines using Natural Language Processing (NLP) and Deep Learning techniques.

The model was trained on nearly **200,000 financial news records** and achieved around **81% test accuracy** using a Bidirectional LSTM architecture with Word2Vec embeddings.

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- NLP
- Word2Vec
- BiLSTM
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 📂 Dataset Information

Dataset used in this project:

- Financial News Headlines Dataset
- Around **200K rows**
- Contains:
  - News headlines/text
  - Sentiment labels

### Dataset Features

- Cleaned financial text data
- Binary sentiment classification
- Large-scale NLP dataset for deep learning

> Dataset is not uploaded because of GitHub file size limitations.

---

## ⚙️ NLP Pipeline

### Data Preprocessing

- Lowercasing
- Removing punctuation
- Removing stopwords
- Tokenization
- Sequence padding

### Embedding Technique

- Custom trained **Word2Vec**
- Embedding dimension: 200

---

## 🧠 Deep Learning Architecture

```python
model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=True
    )
)

model.add(
    Bidirectional(
        LSTM(64)
    )
)

model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))
```

---

## 🧪 Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | ~90% |
| Validation Accuracy | ~81% |
| Test Accuracy | ~81% |

### Observations

- Model started overfitting after a few epochs
- EarlyStopping was used to reduce overfitting
- BiLSTM performed better than traditional ML models

---

## 📈 Traditional ML Baseline

Before deep learning, traditional NLP models were also tested:

- Logistic Regression + TF-IDF
- Linear SVM + TF-IDF

Both achieved around **78% accuracy**.

---

## 📊 Deep Learning Improvement

Using:
- Word2Vec embeddings
- Bidirectional LSTM
- Better preprocessing

Accuracy improved from:
- **78% → 81%**

---

## 💾 Saved Files

- `sentiment_lstm_model.h5` → trained deep learning model
- `tokenizer.pkl` → tokenizer object
- `main.ipynb` → complete notebook

---

## 🧠 Example Predictions

| News Headline | Prediction |
|---|---|
| Stock market crashes after inflation fears | Negative |
| Company profits rise sharply this quarter | Positive |
| Investors are happy with strong earnings | Positive |

---

## 🔥 Key Learning Outcomes

- NLP preprocessing
- Word2Vec embeddings
- Sequence modeling with LSTM
- Bidirectional LSTM
- Overfitting handling
- EarlyStopping
- Deep learning model evaluation
- Financial text sentiment analysis

---

## 📌 Future Improvements

- Use pretrained GloVe embeddings
- Try GRU architecture
- Hyperparameter tuning
- Transformer-based models (BERT)
- Streamlit deployment

---

## 👨‍💻 Author

**Parth Rana**  
AI/ML Engineer | NLP & Deep Learning Enthusiast
