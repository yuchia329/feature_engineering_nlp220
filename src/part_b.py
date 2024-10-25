import os 
import glob 
import pandas as pd
import numpy as np
import torch
import random
import argparse
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from util import calculate_time
from MultinomialNaiveBayes import MultinomialNaiveBayes
from sklearn.model_selection import GridSearchCV

def setSeed():
    seed = 490
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def dataLoader(files):
    df1 = pd.read_csv(files[0])
    df1["rating"] = pd.to_numeric(df1["rating"], downcast='integer', errors='coerce')
    df1Min = df1["rating"].min()
    df2 = pd.read_csv(files[1])
    df2["rating"] = pd.to_numeric(df2["rating"], downcast='integer', errors='coerce')
    df2Min = df2["rating"].min()
    high = df1Min if df1Min > df2Min else df2Min
    df1 = df1._append(df2, ignore_index=True)
    df1["rating"] = df1["rating"].apply(lambda x: 1 if int(x)>=high else 0)
    return df1

def processData(df):
    x = df['review'].values
    y = df['rating'].values
    return x, y

def selectNgram(index):
    options = [{"analyzer":'word', "ngram_range":(1, 1), "max_features":99999, "stop_words": None },
               {"analyzer":'word', "ngram_range":(1, 1), "max_features":99999, "stop_words": "english" },
               {"analyzer":'word', "ngram_range":(1, 2), "max_features":99999, "stop_words": None },
               {"analyzer":'word', "ngram_range":(1, 2), "max_features":99999, "stop_words": "english" },
               {"analyzer":'char', "ngram_range":(4, 4), "max_features":99999, "stop_words": None },
               {"analyzer":'char', "ngram_range":(4, 5), "max_features":99999, "stop_words": None },
               {"analyzer":'char_wb', "ngram_range":(4, 4), "max_features":99999, "stop_words": None },
               {"analyzer":'char_wb', "ngram_range":(4, 5), "max_features":99999, "stop_words": None },
               {"analyzer":'char_wb', "ngram_range":(3, 5), "max_features":99999, "stop_words": None },
               {"analyzer":'char_wb', "ngram_range":(3, 6), "max_features":99999, "stop_words": None}]
    return options[index%len(options)]

def selectClassifier(name={"naive_bayes", "svm", "decision_tree", "self_naive_bayes", "forest", "linear_regression"}):
    if name == "naive_bayes":
        classifier = MultinomialNB()
    elif name == "svm":
        classifier = make_pipeline( LinearSVC(random_state=420, tol=1e-5))
    elif name == "decision_tree":
        classifier = DecisionTreeClassifier(criterion="log_loss", random_state=42)
    elif name == "forest":
        classifier = RandomForestClassifier(max_depth=2, random_state=42)
    elif name == "linear_regression":
        classifier = LogisticRegression(random_state=42)
    else:
        classifier = MultinomialNaiveBayes()
    return classifier

@calculate_time
def trainModel(model, x_train, y_train):
    model.fit(x_train, y_train)

@calculate_time
def predictModel(model, x_test):
    return model.predict(x_test)

def runModel(model, ngramChoice, x, y):
    ngram = selectNgram(ngramChoice)
    print(ngram)
    vectorizer = TfidfVectorizer(stop_words=ngram["stop_words"], analyzer=ngram['analyzer'], ngram_range=ngram['ngram_range'], max_features=ngram['max_features'])
    X = vectorizer.fit_transform(x).toarray()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)
    classifier = selectClassifier(model)
    trainModel(classifier, x_train, y_train)
    y_pred = predictModel(classifier, x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred)
    print(f'none p: {p}  r: {r} f1: {f1}')
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average='macro')
    print(f'macro precision: {p} recall: {r} f1: {f1}')
    print("Classification Report:\n", classification_report(y_test, y_pred))

def executeLoop(x, y, model, features):
    for i in range(features):
        runModel(model, i, x, y)

def main():
    setSeed()
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default='sample_data/part_b',
                        help="Path to the training data CSV files")
    args = parser.parse_args()
    data_dir = args.data_dir
    path = os.getcwd() 
    train_files = glob.glob(os.path.join(path, data_dir, "*.csv"))
    df = dataLoader(train_files)
    x, y = processData(df)
    executeLoop(x, y, "naive_bayes", 0)
    executeLoop(x, y, "svm", 10)
    executeLoop(x, y, "decision_tree", 10)
    executeLoop(x, y, "forest", 10)
    executeLoop(x, y, "linear_regression", 10)

if __name__ == "__main__":
    main()