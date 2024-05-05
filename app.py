import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
#nltk.download('stopwords')
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load JSON data
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Extract intent labels and patterns
labels = []
patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        labels.append(intent['tag'])
        patterns.append(pattern)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)


def get_most_similar_intent(user_input, vectorizer, X, labels):
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, X)
    most_similar_index = np.argmax(similarity_scores)
    return labels[most_similar_index]



def chatbot():
    print("Welcome to the Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        else:
            intent = get_most_similar_intent(user_input, vectorizer, X, labels)
            for intent_data in intents['intents']:
                if intent_data['tag'] == intent:
                    responses = intent_data['responses']
                    print("Chatbot:", np.random.choice(responses))

# Run the chatbot
chatbot()
