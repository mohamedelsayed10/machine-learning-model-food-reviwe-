import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stopwords = set(stopwords.words('english'))
no_stop_words = set(word for word in stopwords 
                          if "n't" in word or 'no' in word)
stopwords = stopwords - no_stop_words
def remove_stop_words(data):
    stop_words =stopwords 
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data
def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")
def preprocssing(data):
    x=data
    x=remove_apostrophe(data)
    x=stemming(data)
    x=remove_punctuation(data)
    x=remove_stop_words(data)
    return x
def raw_test(review, model, vectorizer):
    # Clean Review
    review=preprocssing(review)

    # Embed review using tf-idf vectorizer
    review =vectorizer.transform([review])

    # Predict using your model
    prediction = int(model.predict(review.reshape(1,-1)))
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"
