import pandas as pd
import numpy as np
import torch
import random
import nltk
from sklearn import svm
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
    toDropColumns = list(filter(lambda x: x not in ['review/text', 'review/score'], list(df.columns)))
    df.drop(toDropColumns, axis=1)
    df = df[df['review/score'] != 3]
    x = df['review/text'].values
    y = df['review/score'].values
    return x, y

def downloadNLTK():
    nltk.download('punkt')
    nltk.download('stopwords')



def preProcessText(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetical tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

def main():
    setSeed()
    x, y = processData('small_books_rating.csv')
    reviews_cleaned = [preProcessText(review) for review in x]
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=99999)
    X = vectorizer.fit_transform(reviews_cleaned).toarray()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)
    classifier = BernoulliNB()
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