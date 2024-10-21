import pandas as pd
import numpy as np
import torch
import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

def setSeed():
    seed = 490
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def processData(file_path):
    df = pd.read_csv(file_path)
    toDropColumns = list(filter(lambda x: x not in ['review/text', 'review/score', 'review/summary'], list(df.columns)))
    df = df.drop(toDropColumns, axis=1)
    df['review/score'] = df['review/score'].apply(update_value)
    df = df[df['review/score'] != 0]
    # x = df['review/text'].values
    # print(df.columns)
    df = df.dropna(axis=0)
    # x = df[['review/text','review/summary']].values
    df['review/text'] = df['review/text'] + " " + df['review/summary']
    x = df['review/text'].values
    y = df['review/score'].values
    return x, y

def update_value(x):
    x = int(x)
    if x > 3:
        return 1
    elif x == 3:
        return 0
    else:
        return -1

def downloadNLTK():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(articles)]

def preProcessText(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetical tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

def main():
    setSeed()
    x, y = processData('small_books_rating.csv')
    # reviews_cleaned = [preProcessText(review) for review in x]
    downloadNLTK()
    vectorizer = TfidfVectorizer( analyzer='word', ngram_range=(1, 1), max_features=99999)
    print(x.shape)
    X = vectorizer.fit_transform(x).toarray()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average='macro')
    print(f'micro precision: {p}')
    print(f'micro recall: {r}')
    print(f'micro f1: {f1}')
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    

if __name__ == "__main__":
    main()