# 🧠 AI Fake News Detector (In Progress)

### 🔍 Overview
An AI-powered web application that classifies news articles as **FAKE** or **REAL** using a fine-tuned **BERT** transformer model.  
The project combines **NLP**, **Machine Learning**, and **Full-Stack Web Development** to build a transparent, explainable fake news detection system.

---

### 🚀 Features
- Fine-tuned **BERT (bert-base-uncased)** for binary text classification  
- **Flask REST API** backend serving real-time predictions  
- **Softmax confidence scoring** for interpretability  
- **JSON-based API endpoint** `/predict` for text input and prediction results  
- Pre-processing pipeline using **BertTokenizer** (tokenization, truncation, padding)  
- Model and tokenizer **saved and versioned** for reproducibility (`saved_model/`)  
- Planned **React.js frontend** integration for user-friendly interface  
- Future deployment through **Docker** + **AWS/Heroku**

---

### 🧩 Tech Stack
**Languages & Frameworks:** Python, Flask, PyTorch, React.js (planned)  
**Libraries:** Hugging Face Transformers, Torch, Pandas  
**Tools:** Docker (planned), Git, AWS/Heroku (planned)  

---

### 📂 Project Structure
AI-Fake-News-Detector/
│
├── fake-news-backend/
│ ├── app.py # Flask backend API
│ ├── data/
│ │ ├── train.csv
│ │ ├── val.csv
│ │ ├── test.csv
│ │ ├── train_enc.pkl
│ │ ├── val_enc.pkl
│ │ ├── test_enc.pkl
│ │ └── prepare.py
│ ├── saved_model/ # Fine-tuned model & tokenizer
│ └── requirements.txt
│
└── fake-news-frontend/ (planned)
