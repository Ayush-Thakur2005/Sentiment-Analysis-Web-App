from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        text = request.form["text"]
        text_vectorized = vectorizer.transform([text])
        sentiment_pred = model.predict(text_vectorized)
        sentiment = sentiment_pred[0]
    
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
