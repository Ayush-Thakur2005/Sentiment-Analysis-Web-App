The Sentiment Analysis Web App is an interactive machine learning-powered web application built using Python, Flask, and Scikit-learn. The app classifies text input into three categories: positive, negative, or neutral, helping users analyze the sentiment behind any given text. By utilizing Natural Language Processing (NLP) techniques, this app allows anyone to understand the emotional tone of a text instantly.

The core of the project is a Naive Bayes classifier, trained on a dataset containing labeled text and sentiment data. The application takes user input via a simple web interface built with Flask, processes the text through the model, and displays the predicted sentiment.

Features:
Interactive Web App: Users can input any text and get real-time sentiment analysis.
Sentiment Classification: Classifies text into positive, negative, or neutral categories.
Machine Learning: Uses the Naive Bayes algorithm for sentiment prediction.
User-Friendly Interface: Clean and simple frontend developed with HTML/CSS.
Model Training: Built using a basic sentiment dataset to train the model for accurate predictions.
Technologies Used:
Python for building the sentiment model.
Flask for creating the web application.
Scikit-learn for the machine learning model.
NLTK for text processing.
HTML/CSS for the frontend interface.
How to Use:
Input: Type or paste any text into the provided textarea on the web page.
Submit: After clicking the "Analyze Sentiment" button, the app predicts whether the sentiment of the text is positive, negative, or neutral.
Result: The predicted sentiment is displayed on the same page.
This project demonstrates the application of machine learning in real-time user-facing web applications. You can deploy it on a cloud platform like Heroku to share with others or customize it further.

