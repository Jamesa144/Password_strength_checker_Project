import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(# Path to data set goes here
    , on_bad_lines='skip')

data = data.dropna()

data["strength"] = data["strength"].map({0: "weak", 1: "medium", 2: "strong"})

def word(password):
    return password.split() 

x = np.array(data["password"])
y = np.array(data["strength"])

# Set token_pattern=None to avoid triggering the warning
tfidf = TfidfVectorizer(tokenizer=word, token_pattern=None,max_features=700)

x = tfidf.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

print("Model accuracy:", model.score(xtest, ytest))

# Function to predict password strength
def predict_password_strength(password):
    transformed_password = tfidf.transform([password])  # Transform the user input to match the trained model
    strength = model.predict(transformed_password)
    return strength[0]

# User input for password strength prediction
while True:
    user_password = input("Enter a password to check its strength (or type 'exit' to quit): ")
    if user_password.lower() == 'exit':
        break
    strength = predict_password_strength(user_password)
    print(f"The predicted strength of your password is: {strength}")
