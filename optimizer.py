import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
class DocumentsExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def fit(self, X, y=None):
        if(self.verbose):
            print("Verbose mode on!")
        return self
    
    def transform(self, X, y=None):
        return [" ".join(item['ingredients']) for item in X]

import json
train = json.load(open('./data/train.json'))
de = DocumentsExtractor()
de.fit_transform(train)

tfidf_pipeline = Pipeline([
    ('doc_extractor', DocumentsExtractor()),
    ('tfidf_vectorizer', TfidfVectorizer())
])

tfidf_pipeline.fit_transform(train)


from sklearn.linear_model import LogisticRegression
y_train = [item['cuisine'] for item in train]
lr_pipeline = Pipeline([
    ('tfidf_pipeline', tfidf_pipeline),
    ('lr', LogisticRegression())
])
grid_params = {
  'lr__penalty': ['l1', 'l2'],
  'lr__C': [1, 5, 10],
  'lr__max_iter': [20, 50, 100],
  'tfidf_pipeline__tfidf_vectorizer__max_df': np.linspace(0.1, 1, 10),
  'tfidf_pipeline__tfidf_vectorizer__binary': [True],
}
clf = GridSearchCV(lr_pipeline, grid_params)
clf.fit(train, y_train)
print("Best Score: ", clf.best_score_)
print("Best Params: ", clf.best_params_)