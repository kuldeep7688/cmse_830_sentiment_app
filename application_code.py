import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, MaxAbsScaler
st.set_page_config(layout="wide")

# Title and subtitle
st.write("""
# Twitter Sentiment Analysis
#### Building a Classification Model for Twitter Sentiment Analysis using Sklearn.
"""
)
try:
    df = pd.read_csv('data/train_data.csv')
    print('df loading successful')
except:
    print('Something went wrong in loading data.')

# add introduction about the data 
st.write("""
### Introduction
- Many of us scroll through twitter first thing in the morning. 
- Tweets in Twitter can convey different sentiments to people and it is interesting to build a model that can identify the sentiment of a tweet beforehand.
- This project is about building a classification model on tweets to identify whether they are Positive or Negative.
- The dataset is from Analytics Vidhya website open competitions.
""")

# ADD SEPARATOR
st.write("""
-----
""")


# Feature explanation tab
int_tab_1, int_tab_2 = st.tabs(
    [
        'Text Features', 'Numerical Features',
    ]
)
with int_tab_1:
    st.write("""
    #### There are three types of textual features
    - i will be using straighforward methods like tfidfvectorizer and countvectorizer to extract features from preprocessed text.
    - i used the below mentioned preprocessing steps:
        - removing urls mentioned in the tweets
        - removing usernames mentioned as @ and replacing them with just username
        - removing and replacing all multiple whitespaces with single white space
        - remove topics mentioned as #some_topic. we extract them and use them as a separate feature.
    
    ##### tweet 
    - the tweet is itself a feature, i will be using a tfidf vectorizer to extract features from the original tweet.
    
    ##### topics
    - so i used the hashtags mentioned in a tweet as topics. i just extracted them and again use a tfidfvectorizer to extract features from them.
    
    ##### emojis
    - the emojis present in the tweets are also good features. after extracting i convert them into features using countvectorizer.
    
    """)

with int_tab_2:
    st.write("""
    #### There are five types of numerical features. These features are extracted from text only.
    
    ##### Length of Tweet
    - Basically this is the number of words in a Tweet.

    ##### Number of Topics
    - Total number of hashtags mentioned in the tweet.     
    
    ##### Number of Emojis
    - Total number of emojis present in the tweet.

    ##### Number of Slurrs 
    - Total number of slurrs or abusive words mentioned in the tweet.

    ##### Emoji Score
    - Using scoring dict each emoji is given a score.
    - This score is aggregated sum for all the emojis in the tweet. 
    
    """)

# ADD SEPARATOR
st.write("""
-----
""")

# show label distribution
st.write("""
### Label Distribution And Preparing Train and Test Data
- We wont be using Validation set as we will be using K-fold cross validation for parameter tuning.
- Below is the distribution of the labels in the entire dataset.
- We can see that the dataset is not balanced.
- Next step to split dataframe into train and test. 
""")
chart_1 = alt.Chart(df).mark_bar(size=10).encode(
    x='label:N',
    y='count()',
).properties(
    width=1000,
    height=500
)


# ask for test and valid percentage
st.write("""
### Please select the train and test percentages from the below options.
""")

split_percentages = st.radio(
    'Select the percentage for train and test split ',
    ('Train: 70% | Test: 30%', 'Train: 80% | Test: 20%', 'Train: 90% | Test: 10%', 'Train: 50% | Test: 50%')
)
if split_percentages == 'Train: 70% | Test: 30%':
    test_percentage = 0.3
elif split_percentages == 'Train: 80% | Test: 20%':
    test_percentage = 0.2
elif split_percentages == 'Train: 90% | Test: 10%':
    test_percentage = 0.1
else:
    st.write("""
    ###### Feeling adventurous are we ? :p
    """)
    test_percentage = 0.5

# split data into train, test and validation
train_df, test_df = train_test_split(
    df, test_size=test_percentage, shuffle=True,
    random_state=12, stratify=df['label']
)
st.write(
    'Train Data has : {} tweets and Test Data has : {} tweets. Lets get to modelling.'.format(train_df.shape[0], test_df.shape[0])
)

# ask which text and numerical features to use to users
st.write("""
#### Please select the Numerical Features you are interested in using for model fitting.
""")
all_numeric_features = [
    'num_topics', 'num_emojis', 'length_of_tweet',
    'num_of_slurrs', 'emoji_score'
]
all_textual_features = ['tweet', 'label', 'topics', 'extracted_emojis'] 
numerical_features_to_use = st.multiselect(
    'Numerical Features to use for modelling :',
    all_numeric_features, all_numeric_features
)

all_columns = all_textual_features + numerical_features_to_use 
train_df = train_df[all_columns]
test_df = test_df[all_columns]

# give options for scaling the numerical columns (standard, minmax , max abs)
st.write("""
#### Please select which scaling to use to preprocess numerical features:
""")
scaling_type = st.radio(
    'Select the type of Scaling :',
    ('StandardScaler', 'MinMaxScaler', 'MaxAbsSclaer')
)
if scaling_type == "StandardScaler":
    scaler = StandardScaler()
elif scaling_type == "MinMaxScaler":
    scaler = MinMaxScaler()
else:
    sclaer = MaxAbsScaler()



# ask which classifiers to Fit for Voting Classifier
st.write("""
#### Please select the classifiers to be used inside the Voting Classifer.
- A Voting Classifier takes the output of the all the classifiers that are selected and gives that label as output which has the majority vote among the classifiers.
""")
all_classifiers = [
    'logistic', 'svm', 'random_forest', 'knn', 'naive_bayes'
]
classifiers_to_use = st.multiselect(
    'Numerical Classifiers to use for modelling :',
    all_classifiers, all_classifiers
)

# import pipeline object


# show the pipeline creater 
# show the heyperparameter and ask user to fill
# ask for kfolds to use 
# train model ; show animation

# show best model and parameters
# get results on test set 
# show confusion matrix and sklearn classification report


# optional plot learning curve
# give option to input sentence and get output.