# 🎧 Spotify Song Clustering using Machine Learning

An end-to-end **unsupervised machine learning project** that clusters Spotify songs based on their audio features and deploys the model as an interactive **Streamlit web application**.

---

## 🚀 Project Overview

Spotify hosts millions of songs across different genres, moods, and styles.  
Manually categorizing these songs is not scalable.

This project uses **K-Means clustering** to automatically group songs based on their audio characteristics such as danceability, energy, tempo, and more.

The results are visualized using **PCA (Principal Component Analysis)** and presented through a clean, Spotify-themed web UI.

---

## 🎯 Objectives

- Apply **unsupervised learning** to real-world music data  
- Identify natural groupings of songs  
- Visualize clusters using PCA  
- Build a **production-style Streamlit dashboard**  
- Improve UX with a Spotify-inspired theme  

---

## 📂 Dataset

- **Spotify Audio Features Dataset (April 2019)**
- Each song contains numerical audio attributes extracted from Spotify’s audio analysis engine.

### Features Used:
- Danceability  
- Energy  
- Loudness  
- Tempo  
- Valence  
- Speechiness  
- Liveness  
- Instrumentalness  

---

## 🧠 Machine Learning Pipeline

1. Data Cleaning & Feature Selection  
2. Feature Scaling using `StandardScaler`  
3. Optimal cluster selection using **Elbow Method**  
4. Model training using **K-Means Clustering**  
5. Dimensionality reduction using **PCA**  
6. Interactive visualization & prediction using **Streamlit**

---

## 📊 Clustering Output

The model groups songs into clusters such as:
- High-energy & dance tracks  
- Calm & smooth songs  
- Instrumental / classical music  
- Speech-heavy (rap / spoken content)  
- Balanced pop-style tracks  

---

## 🖥️ Web Application Features

- Sidebar navigation (Prediction | Project Overview | About)
- Two-column feature input layout
- PCA-based cluster visualization
- Mobile-friendly responsive UI
- Spotify-inspired black, green, and white theme

---

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Streamlit**

---


