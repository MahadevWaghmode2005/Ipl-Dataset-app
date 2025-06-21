import streamlit as st   
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Configuration
st.set_page_config(page_title="IPL Match Win Predictor", layout="wide")
st.title("🏏 IPL Logistic Regression Classifier")
st.markdown("Predict whether **team1** wins or not using logistic regression on past IPL data.")

# Load dataset directly from matches.csv
try:
    df = pd.read_csv("matches.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("matches.csv", encoding='ISO-8859-1')
except FileNotFoundError:
    st.error("❌ 'matches.csv' not found. Please place it in the app directory.")
    st.stop()

st.subheader("📊 Raw Dataset Preview")
st.dataframe(df.head())

# Data Cleaning & Target Creation
df = df.dropna()
df['target_win'] = np.where(df['winner'] == df['team1'], 1, 0)

# Visualizations
st.subheader("🎯 Target Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='target_win', data=df, ax=ax1)
st.pyplot(fig1)

st.subheader("🧠 Toss Decision Count")
fig2, ax2 = plt.subplots()
sns.countplot(x='toss_decision', data=df, ax=ax2)
st.pyplot(fig2)

st.subheader("📈 Win by Runs & Wickets")
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['win_by_runs'], bins=30, color='skyblue', ax=ax3a)
sns.histplot(df['win_by_wickets'], bins=30, color='salmon', ax=ax3b)
st.pyplot(fig3)

st.subheader("🏅 Top Players of the Match")
top_players = df['player_of_match'].value_counts().nlargest(10)
fig4, ax4 = plt.subplots()
sns.barplot(x=top_players.values, y=top_players.index, ax=ax4)
st.pyplot(fig4)

# Encode Categorical Variables
categorical_cols = ['Season', 'city', 'team1', 'team2', 'toss_winner', 'toss_decision',
                    'result', 'venue', 'player_of_match', 'umpire1', 'umpire2']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.drop(columns=['id', 'date', 'winner', 'umpire3'], inplace=True, errors='ignore')

# Feature/Target Split
X = df.drop("target_win", axis=1)
y = df["target_win"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

st.subheader("📋 Model Evaluation")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

st.subheader("🧾 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig5, ax5 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax5)
st.pyplot(fig5)

# Feature Importance
st.subheader("📌 Top 10 Important Features")
feature_names = X.columns
importance = model.coef_[0]
coef_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
coef_df = coef_df.reindex(coef_df.Importance.abs().sort_values(ascending=False).index)
fig6, ax6 = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=coef_df.head(10), ax=ax6)
st.pyplot(fig6)

# Prediction Interface
st.subheader("🎯 Make a Prediction")
input_dict = {}
for col in X.columns:
    if "win_by_" in col:
        input_dict[col] = st.number_input(f"{col}", min_value=0, value=0)
    else:
        input_dict[col] = st.selectbox(f"{col}", ["0", "1"], key=col)

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0][1]

if st.button("Predict Outcome"):
    if prediction == 1:
        st.success(f"✅ Team1 is likely to WIN ({pred_proba:.2%} confidence)")
    else:
        st.error(f"❌ Team1 is likely to LOSE ({(1 - pred_proba):.2%} confidence)")
