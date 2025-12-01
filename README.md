# 🧠 AI Fake News Detector (In Progress)

### 🔍 Overview
An AI-powered web application that classifies news articles as **FAKE** or **REAL** using a fine-tuned **BERT** transformer model.  
The project combines **NLP**, **Machine Learning**, and **Full-Stack Web Development** to build a transparent, explainable fake news detection system.

---
###⚡️Localhost Access
---
  Once the application is running locally, you can access the services here:
  Frontend (User Interface): http://localhost:5173
  Backend (API Status): http://localhost:5000/health

### 🚀 Features
-**Fine-tuned BERT Model:** Uses bert-base-uncased for accurate binary text classification.
-**Interactive Frontend:** React.js interface allows users to paste articles and view results instantly.
-**Real-time Analysis:** Flask REST API serves predictions via the /predict endpoint.
-**Confidence Scoring:** Displays a "Veracity" score (probability percentage) calculated via       Softmax logic.
-**Pre-processing Pipeline:** Automated tokenisation, truncation, and padding using BertTokenizer.
-**Reproducibility:** Model and tokeniser are versioned and saved in saved_model/

---

### 🧩 Tech Stack
**Frontend:** React.js, Vite, Axios, CSS3 
**Backend:** Python, Flask, Flask-CORS
**AI/ML:** PyTorch, Hugging Face Transformers, Pandas, Scikit-learn
**Tools:** Git, VS Code


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
└── frontend/ 
  ├── src/
  ├── public/
  └── package.json

  ###🔮 Future Roadmap
---
  **Multi-Modal Detection:** Analyse news URLs/Links, Images, and Uploaded Documents for   authenticity.
  **Explainable AI (XAI):** Add an interpretability layer (SHAP/LIME) to highlight why specific text was flagged as fake.
  **User Feedback Loop:** Allow users to flag incorrect predictions to retrain the model.
  **Deployment:** Containerization via Docker and cloud hosting on AWS or Heroku.

