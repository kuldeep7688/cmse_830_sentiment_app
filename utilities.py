import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer




def preprocess_tweet(tweet):
    #convert the tweet to lower case
    tweet = tweet.lower()
    
    #convert all urls to sting "URL"
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    
    #convert all @username to "AT_USER"
    tweet = re.sub('@[^\s]+','username', tweet)
    
    # coverting "$&@*#" to slur
    tweet = re.sub('$&@*#','profane', tweet)

    #correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    
    #convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet


def extract_topics(tweet):
    pattern = re.compile(r'#([^\s]+)')
    matches = pattern.findall(tweet)
    topic_string = " ".join(matches)
    if len(topic_string) < 1:
        topic_string = "no_topics"
    return topic_string.lower()


def num_topics(tweet):
    pattern = re.compile(r'#([^\s]+)')
    matches = pattern.findall(tweet)
    topic_string = " ".join(matches)
    return float(len(topic_string.split()))


def extract_emojis(tweet):
    pattern = re.compile(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)")
    matches = pattern.findall(tweet)
    emoji_string = " ".join(matches)
    if len(emoji_string) < 1:
        emoji_string = "noemoji"
    return emoji_string.lower()


def num_emojis(string):
    pattern = re.compile(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)")
    matches = pattern.findall(string)
    emoji_string = " ".join(matches)
    return float(len(emoji_string.split()))


def emoji_tokenizer(text):
    return text.split()


# length of tweet
def length_of_tweet(text):
    return len(text.lower().split())


# num of slurrs
def num_of_slurrs(text):
    num_of_slurrs = float(text.count("$&@*#"))
    return num_of_slurrs


# emoji score 
def get_emoji_score(tweet):
    emoji_type_dict = {
        "noemoji": 0,
        ':(': -2,
        ':)': 2,
        ':-(': -2,
        ':-)': 2,
        ':-D': 2,
        ':D': 2,
        ':P': -1,
        ';)': 0,
        ';-)': 0,
        ';D': 0,
        '=(': -2,
        '=)': 2,
        '=D': 2,
        '=P': 0,
        ':-P': -1
    }
    
    pattern = re.compile(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)")
    matches = pattern.findall(tweet)
    emoji_string = " ".join(matches)
    if len(emoji_string) < 1:
        emoji_string = "noemoji"
    score_list = [emoji_type_dict[e] for e in emoji_string.split()]
    return sum(score_list)


def create_pipeline_object(
    scaler, numerical_columns_to_use, classifiers_to_use
):
    # numerical features
    numeric_features = numerical_columns_to_use
    numeric_transformer = Pipeline(
        [
            ("scaler", scaler)
        ]
    )
    
    # text features
    text_features = ['tweet', 'topics', 'extracted_emojis']
    text_transformer = FeatureUnion(
        [
            (
                'tweet_tfidf',
                Pipeline(
                    [
                        (
                            'extract_field',
                            FunctionTransformer(lambda x: x['tweet'], validate=False)
                        ),
                        (
                            'tfidf',
                            TfidfVectorizer()
                        )
                    ]
                )
            ),
            (
                'topic_tfidf',
                Pipeline(
                    [
                        (
                            'extract_field',
                            FunctionTransformer(lambda x: x['topics'], validate=False)
                        ),
                        (
                            'tfidf',
                            TfidfVectorizer()
                        )
                    ]
                )
            ),
            (
                'emoji_tfidf',
                Pipeline(
                    [
                        (
                            'extract_field',
                            FunctionTransformer(lambda x: x['extracted_emojis'], validate=False)
                        ),
                        (
                            'tfidf',
                            TfidfVectorizer()
                        )
                    ]
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("text", text_transformer, text_features),
        ]
    )
    ensemble = VotingClassifier(
        estimators=[
            ('logistic',  LogisticRegression()),
            ("svm", SVC()),
            ("random_forest", RandomForestClassifier()),
            ('knn', KNeighborsClassifier())
        ]
    )
    ppl = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", ensemble)]
    )
    return ppl