from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
#nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Load JSON data and preprocess patterns
with open('intents.json', 'r') as file:
    intents = json.load(file)

labels = []
patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        labels.append(intent['tag'])
        patterns.append(pattern.lower())  # Convert patterns to lowercase for case insensitivity

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

def get_most_similar_intent(user_input, vectorizer, X, labels):
    user_vector = vectorizer.transform([user_input.lower()])  # Convert user input to lowercase
    similarity_scores = cosine_similarity(user_vector, X)
    most_similar_index = np.argmax(similarity_scores)
    return labels[most_similar_index]

# Define request model
class Item(BaseModel):
    message: str

# Define FastAPI endpoint
@app.post("/chatbot/")
async def chatbot(item: Item):
    user_input = item.message
    intent = get_most_similar_intent(user_input, vectorizer, X, labels)
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            responses = intent_data['responses']
            return {"Chatbot": np.random.choice(responses)}

# Serve single HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot</title>
    </head>
    <body>
        <h1>Chatbot</h1>
        <textarea id="user_input" rows="4" cols="50"></textarea><br>
        <button onclick="sendMessage()">Send</button><br>
        <div id="chatbot_response"></div>

        <script>
            async function sendMessage() {
                const user_input = document.getElementById("user_input").value;
                const response = await fetch('/chatbot/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: user_input }),
                });
                const data = await response.json();
                document.getElementById("chatbot_response").innerHTML += "<p>Chatbot: " + data.Chatbot + "</p>";
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
