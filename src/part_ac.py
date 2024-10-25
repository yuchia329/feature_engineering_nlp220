import pandas as pd
import numpy as np
import torch
import random
import argparse
import nltk
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels
from util import calculate_time
from MultinomialNaiveBayes import MultinomialNaiveBayes

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
    df = df.dropna(axis=0)
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

def preProcessText(text, isalpha=False, stopwords=False):
    # downloadNLTK()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha()] if isalpha else tokens  # Remove non-alphabetical tokens
    tokens = [word for word in tokens if word not in stop_words] if stopwords else tokens  # Remove stopwords
    return " ".join(tokens)

def plotMatrix(y_true, y_pred, classifier, filename):
    # print(getattr(classifier, 'classes_', None))
    if getattr(classifier, 'classes_', None) is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = classifier.classes_
    print("labels: ", labels)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot().figure_.savefig(filename)

def feature1(x):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=99999)
    x = [preProcessText(review, isalpha=False, stopwords=False) for review in x]
    return x, vectorizer

def feature2(x):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=99999)
    x = [preProcessText(review, isalpha=True, stopwords=True) for review in x]
    return x, vectorizer

def feature3(x):
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), analyzer='word', ngram_range=(1, 1), max_features=99999)
    x = [preProcessText(review, isalpha=False, stopwords=False) for review in x]
    return x, vectorizer

def selectClassifier(name={"naive_bayes", "svm", "decision_tree", "self_naive_bayes"}):
    if name == "naive_bayes":
        classifier = MultinomialNB()
    elif name == "svm":
        classifier = make_pipeline( LinearSVC(random_state=420, tol=1e-5))
    elif name == "decision_tree":
        classifier = DecisionTreeClassifier(criterion="gini", random_state=42)
    else:
        classifier = MultinomialNaiveBayes()
    return classifier

@calculate_time
def trainModel(model, x_train, y_train):
    model.fit(x_train, y_train)

@calculate_time
def predictModel(model, x_test):
    return model.predict(x_test)

@calculate_time
def runModel(train_file, plot_confusion, name, feature):
    
    x, y = processData(train_file)
    if feature == 1:
        x, vectorizer = feature1(x)
    elif feature == 2:
        x, vectorizer = feature2(x)
    else:
        x, vectorizer = feature3(x)
    X = vectorizer.fit_transform(x).toarray()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    classifier = selectClassifier(name)
    
    trainModel(classifier, x_train, y_train)
    y_pred = predictModel(classifier, x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred)
    print(f'none p: {p}  r: {r} f1: {f1}')
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average='macro')
    print(f'macro precision: {p} recall: {r} f1: {f1}')
    print("Classification Report:\n", classification_report(y_test, y_pred))
    if plot_confusion:
        filename = name + '_feature' + str(feature)
        plotMatrix(y_test, y_pred, classifier, filename)

def main():
    setSeed()
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str, default='sample_data/part_ac/small_books_rating_mini.csv', nargs='?',
                        help="Path to the training data CSV file")
    parser.add_argument("plot_confusion", type=bool, default=False, nargs='?',
                        help="Whether to plot confusion matrix")
    args = parser.parse_args()
    train_file = args.train_file
    plot_confusion = args.plot_confusion

    runModel(train_file, plot_confusion, 'naive_bayes', 1)
    runModel(train_file, plot_confusion, 'naive_bayes', 2)
    runModel(train_file, plot_confusion, 'naive_bayes', 3)

    runModel(train_file, plot_confusion, 'svm', 1)
    runModel(train_file, plot_confusion, 'svm', 2)
    runModel(train_file, plot_confusion, 'svm', 3)

    runModel(train_file, plot_confusion, 'decision_tree', 1)
    runModel(train_file, plot_confusion, 'decision_tree', 2)
    runModel(train_file, plot_confusion, 'decision_tree', 3)

    runModel(train_file, plot_confusion, 'self_naive_bayes', 1)
    runModel(train_file, plot_confusion, 'self_naive_bayes', 2)
    runModel(train_file, plot_confusion, 'self_naive_bayes', 3)
    
    

if __name__ == "__main__":
    main()