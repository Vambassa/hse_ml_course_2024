import click
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def stemmer_deleter(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    word_tokens = word_tokenize(' '.join(stemmed_words))
    filtered_text = [word for word in word_tokens if word not in stop_words]

    return ' '.join(filtered_text)


def preprocess_data(df):
    df['title_text'] = df['title'] + ' ' + df['text']
    df['title_text'] = df['title_text'].apply(stemmer_deleter)
    df['rating'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)

    return df


def calculate_score(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    score = f1_score(y_test, preds)
    click.echo(f'F1-score: {score:.3f}, test size = {len(y_test)}')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--data', type=click.Path(exists=True), help='Path to the train dataset')
@click.option('--test', type=click.Path(), help='Path to the test dataset')
@click.option('--split', type=float, help='Split size for testing')
@click.option('--model', type=click.Path(), help='Path to the trained model')
@click.option('--seed', type=int, default=42, help='Random state')
def train(data, test, split, model, seed):

    if not os.path.exists(data):
        raise FileNotFoundError('This data file does not exist')
    
    df = pd.read_csv(data)
    df = preprocess_data(df)

    X_train = df['title_text']
    y_train = df['rating']

    if test:

        test_df = pd.read_csv(test)
        test_df = preprocess_data(test_df)
        X_test = test_df['title_text']
        y_test = test_df['rating']

    elif split:

        X_train, X_test, y_train, y_test = train_test_split(
            df['title_text'],
            df['rating'],
            test_size=split,
            random_state=seed)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('model', LogisticRegression(random_state=seed, max_iter=100))
    ])

    pipeline.fit(X_train, y_train)

    if test or split:
        calculate_score(pipeline, X_test, y_test)

    with open(model, 'wb') as f:
        pickle.dump(pipeline, f)


@cli.command()
@click.option('--model', type=click.Path(exists=True), help='Path to the trained model')
@click.option('--data', help='Path to the test dataset')
def predict(model, data):

    with open(model, 'rb') as f:
        pipeline = pickle.load(f)

    if data.endswith('.csv'):
        df = pd.read_csv(data)
        df = preprocess_data(df)
        X_test = df['title_text']

        preds = pipeline.predict(X_test)

        for pred in preds:
            click.echo(pred)
            
    else:
        
        processed_text = stemmer_deleter(data)
        pred = pipeline.predict([processed_text])

        click.echo(pred[0])


if __name__ == '__main__':
    cli()
