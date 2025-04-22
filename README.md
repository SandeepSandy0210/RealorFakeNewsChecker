# 📰 Fake‑News Classifier

An end‑to‑end **Streamlit** web app that detects whether a news article is *real* or *fake* using a TF‑IDF + Multinomial Naive Bayes model.

<p align="center">
  <a href="https://realorfakenewschecker.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit" />
  </a>
  <img src="https://img.shields.io/github/languages/top/SandeepSandy0210/RealorFakeNewsChecker" />
  <img src="https://img.shields.io/github/license/SandeepSandy0210/RealorFakeNewsChecker" />
</p>

---

## ✨ Demo

<div align="center">
  <img src="docs/demo.gif" width="650"/>
</div>

---

## 📂 Repository structure

├── app.py # Streamlit front‑end 
├── file1.py # (optional) script to train the model locally 
├── model.pkl # Trained MultinomialNB model 
├── vectorizer.pkl # Fitted TfidfVectorizer 
├── requirements.txt # Python dependencies 
└── README.md

> **Note:** The original `Fake.csv` and `True.csv` datasets are large;  
> to keep the repo lightweight they are **not** stored here.  
> Train locally once (`python file1.py`) to regenerate the pickles.

---

## 🚀 Quick start

### 1. Clone and create a virtual env

```bash```
git clone https://github.com/your-username/your-repo.git
cd your-repo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


2. Run locally
bash
Copy
Edit
streamlit run app.py
Visit http://localhost:8501 in your browser.

3. ☁️ Deploy on Streamlit Community Cloud
Fork this repo to your GitHub.

Go to https://streamlit.io/cloud → “New app”.

Select your‑username / your‑repo, branch main, and app.py as the entry point.

Click Deploy.
Your app will be live at
https://your-repo--main--your-username.streamlit.app

