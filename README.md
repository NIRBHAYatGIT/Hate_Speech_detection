# Hate Speech Detection System

A Machine Learning-based web application that detects hate speech in text using Natural Language Processing (NLP). The system classifies user input as Hate Speech or Non-Hate Speech through a trained Logistic Regression model and provides predictions via a Flask web interface.

---

## 📌 Project Overview

Social media platforms generate millions of comments and posts every day. Identifying harmful or offensive content manually is difficult and time-consuming. This project automates the detection of hate speech using Machine Learning and Natural Language Processing techniques.

---

## ✨ Features

- Detects hate speech from user-entered text
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Logistic Regression classifier
- Flask-based web application
- Simple and user-friendly interface
- Fast prediction results

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- NLTK

---

## 📂 Project Structure

```
Hate_Speech_detection/
│
├── app.py
├── requirements.txt
├── model/
├── templates/
├── static/
├── dataset/
├── notebooks/
└── README.md
```

---

## ⚙️ Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. TF-IDF Vectorization
5. Model Training
6. Model Evaluation
7. Flask Deployment

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/NIRBHAYatGIT/Hate_Speech_detection.git
```

### Go to Project Folder

```bash
cd Hate_Speech_detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## 📊 Dataset

The model is trained on a labeled hate speech dataset containing examples of hateful and non-hateful text.

Dataset preprocessing includes:

- Lowercase conversion
- Removing punctuation
- Stopword removal
- Tokenization
- Lemmatization

---

## 📈 Model

Algorithm Used:

- Logistic Regression

Feature Extraction:

- TF-IDF Vectorizer

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 🎯 Future Improvements

- Deep Learning (LSTM/BERT)
- Multi-language hate speech detection
- REST API support
- Real-time social media integration
- Improved UI/UX
- Model performance optimization

---

## 👨‍💻 Author

**Nirbhay Lamba**

MCA Student | NIET Greater Noida

GitHub: https://github.com/NIRBHAYatGIT

---

## 📄 License

This project is created for educational and academic purposes.
