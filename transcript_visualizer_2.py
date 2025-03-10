import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import re
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import entropy

# 🚀 Load the conversation transcript
file_path = "skander_transcript.txt"
with open(file_path, "r", encoding="utf-8") as file:
    conversation_text = file.read()

# 🚀 Extracting speaker labels and messages
user_messages, chatgpt_messages = [], []
timestamps, current_message = [], []
current_speaker = None

for line in conversation_text.split("\n"):
    if line.startswith("You said:"):
        if current_speaker == "ChatGPT":
            chatgpt_messages.append(" ".join(current_message))
        elif current_speaker == "User":
            user_messages.append(" ".join(current_message))
        current_speaker = "User"
        current_message = []
    elif line.startswith("ChatGPT said:"):
        if current_speaker == "User":
            user_messages.append(" ".join(current_message))
        elif current_speaker == "ChatGPT":
            chatgpt_messages.append(" ".join(current_message))
        current_speaker = "ChatGPT"
        current_message = []
    else:
        current_message.append(line)

# Append last message
if current_speaker == "User":
    user_messages.append(" ".join(current_message))
elif current_speaker == "ChatGPT":
    chatgpt_messages.append(" ".join(current_message))

# 🚀 Sentiment & Cognitive Analysis
def analyze_sentiment(messages):
    sentiments = [TextBlob(msg).sentiment.polarity for msg in messages]
    subjectivity = [TextBlob(msg).sentiment.subjectivity for msg in messages]
    return sentiments, subjectivity

user_sentiments, user_subjectivity = analyze_sentiment(user_messages)
chatgpt_sentiments, chatgpt_subjectivity = analyze_sentiment(chatgpt_messages)

# 🚀 Conversational Flow Metrics
response_lengths = np.array([len(msg.split()) for msg in user_messages])
chatgpt_response_lengths = np.array([len(msg.split()) for msg in chatgpt_messages])

# 🚀 Lexical Complexity & Information Entropy
words = re.findall(r'\b\w+\b', conversation_text.lower())
word_counts = Counter(words)
lexical_diversity = len(set(words)) / len(words)
word_frequencies = np.array(list(word_counts.values())) / sum(word_counts.values())
dialogue_entropy = entropy(word_frequencies)

# 🚀 Conversational Power Dynamic
def calculate_power_index(user_lens, ai_lens):
    total_words = user_lens.sum() + ai_lens.sum()
    return user_lens.sum() / total_words if total_words else 0.5

power_dynamic_index = calculate_power_index(response_lengths, chatgpt_response_lengths)

# 🚀 Conversational Topic Analysis
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(user_messages + chatgpt_messages).toarray()
topic_distances = [cosine(tfidf_matrix[i], tfidf_matrix[i+1]) if i+1 < len(tfidf_matrix) else 0 for i in range(len(tfidf_matrix)-1)]

# 🚀 Detecting Escalation Cascades
escalation_indices = [i for i in range(1, len(user_sentiments)) if user_sentiments[i] < user_sentiments[i-1] and user_sentiments[i] < -0.3]

# 🚀 PLOT 1: Emotional Divergence Over Time
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=list(range(len(user_sentiments))), y=user_sentiments,
    mode='lines+markers', name='User Sentiment', line=dict(color='red', width=2)
))
fig1.add_trace(go.Scatter(
    x=list(range(len(chatgpt_sentiments))), y=chatgpt_sentiments,
    mode='lines+markers', name='ChatGPT Sentiment', line=dict(color='blue', width=2, dash='dash')
))
fig1.update_layout(
    title="Emotional Divergence Over Time",
    xaxis_title="Message Order",
    yaxis_title="Sentiment Polarity",
    template="plotly_dark"
)

# 🚀 PLOT 2: Lexical Complexity & Dialogue Entropy
fig2 = go.Figure()
fig2.add_trace(go.Indicator(
    mode="gauge+number",
    value=lexical_diversity,
    title={"text": "Lexical Diversity Score"},
    gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "purple"}}
))
fig2.add_trace(go.Indicator(
    mode="gauge+number",
    value=dialogue_entropy,
    title={"text": "Dialogue Entropy (Word Variability)"},
    gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "orange"}}
))
fig2.update_layout(title="Lexical Complexity & Information Entropy", template="plotly_dark")

# 🚀 PLOT 3: Conversational Power Dynamic
fig3 = go.Figure()
fig3.add_trace(go.Indicator(
    mode="gauge+number",
    value=power_dynamic_index,
    title={"text": "Conversational Power Dynamic"},
    gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "cyan"}}
))
fig3.update_layout(template="plotly_dark")

# 🚀 PLOT 4: Topic Transition & Escalation Analysis
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=list(range(len(topic_distances))), y=topic_distances,
    mode='lines+markers', name='Topic Transition Dissonance', line=dict(color='purple', width=2)
))
if escalation_indices:
    fig4.add_trace(go.Scatter(
        x=escalation_indices, y=[topic_distances[i] for i in escalation_indices],
        mode='markers', marker=dict(color='red', size=10, symbol='triangle-up'),
        name="Escalation Points"
    ))
fig4.update_layout(
    title="Topic Shifts & Escalation Cascades",
    xaxis_title="Message Order",
    yaxis_title="Cosine Distance (Higher = Bigger Topic Shift)",
    template="plotly_dark"
)

# 🚀 Display all plots
fig1.show()
fig2.show()
fig3.show()
fig4.show()