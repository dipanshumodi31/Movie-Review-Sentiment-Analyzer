
# 🎬 Movie Sentiment Analyzer  

![Project Banner](./assets/banner.png)  
A complete end-to-end sentiment analysis project on movie reviews.  

---

## 📖 Table of Contents  
1. [Overview](#overview)  
2. [Objectives](#objectives)  
3. [Workflow](#workflow)  
   - [1. Data Loading](#1-data-loading)  
   - [2. Data Preprocessing](#2-data-preprocessing)  
   - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)  
   - [4. Feature Engineering](#4-feature-engineering)  
   - [5. Model Training & Evaluation](#5-model-training--evaluation)  
   - [6. Error Analysis](#6-error-analysis)  
   - [7. Deployment Demo](#7-deployment-demo)  
4. [Visualizations](#visualizations)  
5. [Results](#results)  
6. [Applications](#applications)  
7. [Future Improvements](#future-improvements)  
8. [Credits](#credits)  

---

## 📝 Overview  
This project demonstrates a *sentiment analysis pipeline* for IMDB movie reviews using *Natural Language Processing (NLP)* and *Machine Learning (ML)* techniques.  

It covers the entire workflow:  
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Feature engineering using BoW & TF-IDF  
- Model building and evaluation  
- Error analysis for misclassifications  
- A simple deployment demo for real-world predictions  

---

## 🎯 Objectives  
- Develop a *robust sentiment classifier* for movie reviews.  
- Compare multiple ML models on BoW and TF-IDF features.  
- Evaluate models using classification metrics and visualization techniques.  
- Showcase the project in a *professional, industry-ready format*.  

---

## ⚙ Workflow  

### *1. Data Loading*  
- Dataset: IMDB Movie Reviews (~5,000 reviews sample).  
- Balanced positive/negative sentiment classes.  

### *2. Data Preprocessing*  
- Tokenization, stopword removal, stemming.  
- Lowercasing and normalization.  
- Constructed n-grams for contextual meaning.  

### *3. Exploratory Data Analysis (EDA)*  
- Word frequency analysis.  
- WordClouds for *Positive* vs *Negative* reviews.  
- Distribution of tokens and sentiment-indicative words.  

### *4. Feature Engineering*  
- Bag-of-Words (BoW).  
- Term Frequency–Inverse Document Frequency (TF-IDF).  
- Comparison between feature representations.  

### *5. Model Training & Evaluation*  
Algorithms trained & tested:  
- Logistic Regression ✅  
- Naive Bayes (BernoulliNB) ✅  
- Support Vector Machine (SVC) ✅  
- Random Forest ✅  

*Metrics Evaluated:*  
- Accuracy, Precision, Recall, F1-score  
- Confusion Matrix  
- ROC Curve & AUC  

### *6. Error Analysis*  
- Misclassified reviews explored.  
- Identified challenges: sarcasm, negations, and mixed opinions.  

### *7. Deployment Demo*  
- Simple prototype where users input a custom review.  
- Model outputs predicted sentiment (Positive / Negative).  

---

## 📊 Visualizations  

Here are the key visuals from the project (placeholders included):  

1. *WordCloud Comparison* of Positive vs Negative Reviews  
   ![WordCloud Comparison](./assets/wordcloud_placeholder.png)  

2. *Model Performance Comparison*  
   ![Model Comparison](./assets/models_placeholder.png)  

3. *Confusion Matrix* for Best Model  
   ![Confusion Matrix](./assets/confusion_matrix_placeholder.png)  

4. *ROC Curve & AUC Plot*  
   ![ROC Curve](./assets/roc_curve_placeholder.png)  

5. *Deployment Demo Screenshot*  
   ![Deployment Demo](./assets/demo_placeholder.png)  

---

## 🏆 Results  
- *Best Performing Model:* Logistic Regression with *TF-IDF features*.  
- *Strengths:* High accuracy, interpretable, and computationally efficient.  
- *Limitations:* Struggles with sarcasm, negations, and nuanced reviews.  

---

## 🌍 Applications  
- Movie & product review platforms (IMDB, Amazon, Flipkart).  
- Customer feedback monitoring (hospitality, retail, SaaS).  
- Social media opinion mining (Twitter, Reddit).  
- Recommendation systems & automated moderation.  

---

## 🚀 Future Improvements  
- Incorporate *Deep Learning models* (LSTM, BERT).  
- Use *pretrained embeddings* (Word2Vec, GloVe, Transformer-based).  
- Improve sarcasm & context handling with contextual embeddings.  
- Build a *web app or API service* for real-world deployment.  

---

## 👨‍💻 Credits  
Created by *Dipanshu Modi*  

🔗 Connect with me on:  
- [LinkedIn](https://www.linkedin.com/in/dipanshu-modi-75bb57278/)  
- [GitHub](https://github.com/dipanshumodi31)  

---
