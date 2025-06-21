import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Setup
st.set_page_config(page_title="IPL Win Predictor (Simple)", layout="wide")
st.title("🏏 Simple IPL Logistic Regression Predictor")

# Load Data
try:
    df = pd.read_csv("matches.csv", encoding='utf-8')
except:
    df = pd.read_csv("matches.csv", encoding='ISO-8859-1')

# Drop missing values
df.dropna(inplace=True)

# Create target column
df['target_win'] = np.where(df['winner'] == df['team1'], 1, 0)

# Feature engineering: Simple binary flags for selected features
df['toss_decision_field'] = np.where(df['toss_decision'] == 'field', 1, 0)
df['team1_MI'] = np.where(df['team1'] == 'Mumbai Indians', 1, 0)
df['team2_CSK'] = np.where(df['team2'] == 'Chennai Super Kings', 1, 0)

# Selected Features
selected_features = ['win_by_runs', 'win_by_wickets', 'toss_decision_field', 'team1_MI', 'team2_CSK']
X = df[selected_features]
y = df['target_win']

# Train/Test Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📋 Model Evaluation (on 5 features)")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title("Confusion Matrix")
st.pyplot(fig1)

# Prediction Section
st.subheader("🎯 Make a Simple Prediction (5 features)")
win_by_runs = st.number_input("Win by Runs", min_value=0, value=0)
win_by_wickets = st.number_input("Win by Wickets", min_value=0, value=0)
toss_field = st.selectbox("Toss Decision", ["bat", "field"]) == "field"
team1_mi = st.selectbox("Is Team1 Mumbai Indians?", ["No", "Yes"]) == "Yes"
team2_csk = st.selectbox("Is Team2 Chennai Super Kings?", ["No", "Yes"]) == "Yes"

# Prepare input
input_data = pd.DataFrame([{
    'win_by_runs': win_by_runs,
    'win_by_wickets': win_by_wickets,
    'toss_decision_field': int(toss_field),
    'team1_MI': int(team1_mi),
    'team2_CSK': int(team2_csk)
}])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
confidence = model.predict_proba(input_scaled)[0][1]

if st.button("Predict Outcome"):
    if prediction == 1:
        st.success(f"✅ Team1 is likely to WIN ({confidence:.2%} confidence)")
    else:
        st.error(f"❌ Team1 is likely to LOSE ({(1 - confidence):.2%} confidence)")
