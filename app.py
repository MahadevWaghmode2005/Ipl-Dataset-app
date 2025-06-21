import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Configuration
st.set_page_config(page_title="IPL Win Predictor (Minimal)", layout="wide")
st.title("🏏 IPL Win Predictor ")

# Load dataset
try:
    df = pd.read_csv("matches.csv", encoding='utf-8')
except:
    df = pd.read_csv("matches.csv", encoding='ISO-8859-1')

df = df.dropna()
df['target_win'] = np.where(df['winner'] == df['team1'], 1, 0)

# Feature Engineering (only these 4 features)
df['toss_decision_field'] = np.where(df['toss_decision'] == 'field', 1, 0)
df['team1_is_MI'] = np.where(df['team1'] == 'Mumbai Indians', 1, 0)
df['team2_is_CSK'] = np.where(df['team2'] == 'Chennai Super Kings', 1, 0)

selected_features = ['win_by_runs', 'win_by_wickets', 'toss_decision_field', 'team1_is_MI', 'team2_is_CSK']
X = df[selected_features]
y = df['target_win']

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Show Evaluation
st.subheader("📊 Model Accuracy")
y_pred = model.predict(X_test)
st.write("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
st.pyplot(fig)

# Prediction Input Section
st.subheader("🎯 Predict Outcome (Only 4 Inputs)")

# Get Inputs
win_by_runs = st.number_input("Win by Runs", min_value=0, value=0)
win_by_wickets = st.number_input("Win by Wickets", min_value=0, value=0)
toss_field = st.selectbox("Did the team choose to field?", ["No", "Yes"]) == "Yes"
team1_is_mi = st.selectbox("Is Team1 Mumbai Indians?", ["No", "Yes"]) == "Yes"
team2_is_csk = st.selectbox("Is Team2 Chennai Super Kings?", ["No", "Yes"]) == "Yes"

# Prepare input
user_input = pd.DataFrame([{
    'win_by_runs': win_by_runs,
    'win_by_wickets': win_by_wickets,
    'toss_decision_field': int(toss_field),
    'team1_is_MI': int(team1_is_mi),
    'team2_is_CSK': int(team2_is_csk)
}])
user_input_scaled = scaler.transform(user_input)

# Predict
if st.button("Predict Match Result"):
    prediction = model.predict(user_input_scaled)[0]
    confidence = model.predict_proba(user_input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Team1 is likely to WIN ({confidence:.2%} confidence)")
    else:
        st.error(f"❌ Team1 is likely to LOSE ({(1 - confidence):.2%} confidence)")
