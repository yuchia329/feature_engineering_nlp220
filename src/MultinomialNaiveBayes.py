import numpy as np
from collections import defaultdict

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_prob = {}  # P(C)
        self.feature_prob = defaultdict(lambda: defaultdict(float))  # P(X_i | C)
        self.class_count = defaultdict(int)  # Count of samples per class
        self.feature_count = defaultdict(lambda: defaultdict(int))  # Count of features per class
        self.total_features = defaultdict(int)  # Total count of all features per class


    def fit(self, X, y):
        """
        Fit the Multinomial Naive Bayes classifier.
        X: List of document-word frequency arrays (bag-of-words)
        y: List of labels corresponding to each document
        """
        # Calculate prior probabilities P(C)
        total_samples = len(y)
        for i, label in enumerate(y):
            self.class_count[label] += 1
            self.total_features[label] += sum(X[i])  # Total words in class

            # Update feature counts
            for j, count in enumerate(X[i]):
                if count > 0:
                    self.feature_count[label][j] += count

        # Calculate P(C)
        for label, count in self.class_count.items():
            self.class_prob[label] = count / total_samples

        # Calculate P(X_i | C) with Laplace smoothing
        for label in self.class_count:
            total_feature_count = self.total_features[label]
            num_features = len(X[0])
            for j in range(num_features):
                # Laplace smoothing
                self.feature_prob[label][j] = (self.feature_count[label][j] + 1) / (total_feature_count + num_features)

    def predict(self, X):
        """
        Predict the label for a new document.
        X: Document-word frequency array (bag-of-words)
        """
        predictions = []
        for doc in X:
            class_scores = {}
            for label in self.class_prob:
                # Initialize score with log(P(C))
                score = np.log(self.class_prob[label])

                # Add log(P(X_i | C)) for each feature
                for j, count in enumerate(doc):
                    if count > 0:
                        score += count * np.log(self.feature_prob[label][j])

                class_scores[label] = score

            # Choose the class with the highest score
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)

        return predictions

    def score(self, X, y):
        """
        Calculate accuracy on the test set.
        X: List of document-word frequency arrays (bag-of-words)
        y: List of true labels
        """
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)
