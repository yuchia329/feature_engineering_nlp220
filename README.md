# Three Models Study

The project conducts thorough study on performance of Naive Bayes, SVM, decision tree Models with various feature sets. It also includes random forest, and logistic regression for shallow comparison. Our evluation is based on their performance on binary classification problem.

## Installation

Install packages

```
pip install -r requirements.txt
```

All the final files are in src.

### Part A & C

Execute part_ac.py with path to "train_file".
A sample dataset from the complete dataset is included in the repo. Do provide "tran_file" and "plot_confusion" value for 1st and 2nd arguments.

```
python src/part_ac.py sample_data/part_ac/small_books_rating_mini.csv False
```

### Part B

First, write individual record files to 2 classified CSV. Specify the positibe and negative reviews' folder directory and save_file directory for 1st, 2nd, and 3rd arguments.

We provide sample data set in sample_data folder for small batch validation.

```
python src/part_b_convert.py sample_data/part_b/raw/pos sample_data/part_b/raw/neg sample_data/part_b
```

Then, execute part_b.py for model training and testing. Specify data_dir for where the csv files are saved from the previous command

The script will evaluate 5 models with 10 different ngram features. Modify main() function in part_b.py for different scope of measurement.

```
python src/part_b.py sample_data/part_b
```
