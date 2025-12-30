import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Spotify Song Clustering",
    page_icon="🎧",
    layout="wide"
)

# ================= CSS (Spotify Theme) =================
st.markdown("""
<style>
.stApp { background-color: #000000; color: white; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #121212; }

/* Sidebar nav */
div[role="radiogroup"] > label {
    background-color: #121212;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 6px;
    color: white;
}

/* Active tab */
div[role="radiogroup"] > label[data-checked="true"] {
    background-color: #1DB954 !important;
    color: black !important;
    font-weight: bold;
}

/* Hover */
div[role="radiogroup"] > label:hover {
    background-color: #1DB954;
    color: black;
}

/* Header */
.header-box {
    background-color: #1DB954;
    padding: 22px;
    border-radius: 10px;
    text-align: center;
    color: black;
}

/* Result box */
.metric {
    background-color: #121212;
    border-left: 5px solid #1DB954;
    padding: 20px;
    border-radius: 8px;
    font-size: 18px;
}

/* Mobile */
@media (max-width: 768px) {
    .header-box h1 { font-size: 22px; }
    .metric { font-size: 16px; }
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
kmeans = pickle.load(open("kmeans.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ================= SIDEBAR =================
st.sidebar.title("🎧 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🎵 Prediction", "📊 Project Overview", "👤 About"]
)

# ================= HEADER =================
st.markdown("""
<div class="header-box">
    <h1>🎧 Spotify Song Clustering</h1>
    <p>Machine Learning based music clustering using Spotify audio features</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= CLUSTER MEANING =================
cluster_meaning = {
    0: "🔥 High energy & dance tracks",
    1: "🌿 Calm & smooth songs",
    2: "🎼 Instrumental / classical music",
    3: "🎤 Speech-heavy (rap / spoken)",
    4: "🎧 Balanced pop-style songs"
}

# ======================================================
# 🎵 PREDICTION PAGE
# ======================================================
if page == "🎵 Prediction":
    st.subheader("🎵 Song Cluster Prediction")

    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
        tempo = st.slider("Tempo", 50.0, 200.0, 120.0)

    with col2:
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)

    input_data = np.array([[danceability, energy, loudness, tempo,
                            valence, speechiness, liveness, instrumentalness]])

    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric">
        🎶 <b>Predicted Cluster:</b> {cluster}<br><br>
        {cluster_meaning.get(cluster)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 PCA Cluster Visualization")

    df = pd.read_csv("SpotifyAudioFeaturesApril2019.csv")

    features = [
        'danceability','energy','loudness','tempo',
        'valence','speechiness','liveness','instrumentalness'
    ]

    X = df[features].dropna()
    X_scaled = scaler.transform(X)

    df = df.loc[X.index]
    df['Cluster'] = kmeans.predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    user_pca = pca.transform(input_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))

    for c in sorted(df['Cluster'].unique()):
        ax.scatter(
            df[df['Cluster'] == c]['PCA1'],
            df[df['Cluster'] == c]['PCA2'],
            alpha=0.5,
            label=f"Cluster {c}"
        )

    ax.scatter(
        user_pca[0, 0],
        user_pca[0, 1],
        color='red',
        marker='*',
        s=300,
        label='Your Song'
    )

    ax.set_title("Spotify Song Clusters (PCA)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()

    st.pyplot(fig, use_container_width=True)

# ======================================================
# 📊 PROJECT OVERVIEW (DETAILED)
# ======================================================
elif page == "📊 Project Overview":
    st.subheader("📊 Project Overview")

    st.markdown("""
    ### 🔍 Problem Statement
    Spotify hosts millions of songs across genres and moods. Manual classification
    is not scalable. This project uses **unsupervised machine learning** to
    automatically group songs based on their audio characteristics.

    ### 📂 Dataset
    **Spotify Audio Features Dataset (April 2019)** containing numerical
    attributes extracted directly from Spotify’s audio analysis engine.

    ### 🎚️ Features Used
    - **Danceability** – How suitable a track is for dancing  
    - **Energy** – Intensity and activity  
    - **Loudness** – Overall volume (dB)  
    - **Tempo** – Speed of the song (BPM)  
    - **Valence** – Musical positivity  
    - **Speechiness** – Presence of spoken words  
    - **Liveness** – Audience presence  
    - **Instrumentalness** – Vocal vs instrumental content  

    ### 🧠 Machine Learning Pipeline
    1. Data Cleaning & Feature Selection  
    2. Standard Scaling  
    3. K-Means Clustering  
    4. Optimal K selection using Elbow Method  
    5. PCA for 2D Visualization  

    ### 📈 Outcome
    Songs are grouped into distinct clusters such as energetic tracks,
    calm songs, instrumental music, and speech-heavy content.

    ### 🛠️ Tech Stack
    Python · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit
    """)

# ======================================================
# 👤 ABOUT (STRONG)
# ======================================================
elif page == "👤 About":
    st.subheader("👤 About")

    st.markdown("""
    **Developer:** Kamalesh  
    **Role:** Aspiring Data Scientist  

    ### 🎯 Objective
    To build end-to-end machine learning projects using real-world datasets,
    focusing on model development, evaluation, visualization, and deployment.

    ### 🚀 Skills Demonstrated
    - Unsupervised Machine Learning  
    - Feature Engineering & Scaling  
    - K-Means Clustering & PCA  
    - Data Visualization  
    - Streamlit Web App Deployment  

    ### 💡 Why This Project?
    This project demonstrates how machine learning can be used to uncover
    hidden patterns in large-scale music data and present insights through
    an interactive dashboard.

    ### 🔗 Connect
    - GitHub: https://github.com/kamaleshcr7 
    - LinkedIn: www.linkedin.com/in/kamalesh-v-a1504a33a 
    """)

# ================= FOOTER =================
st.markdown("""
<hr>
<p style='text-align:center;'>Built with ❤️ using Streamlit & Machine Learning</p>
""", unsafe_allow_html=True)
