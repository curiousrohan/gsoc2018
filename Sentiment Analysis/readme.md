# Sentiment analysis of GOP debate
To demonstrate the proof of work for this project, I downloaded the GOP debate dataset from Kaggle. It consists of tweets and labelled sentiments for the 2016 GOP debate. My goal was to build a classifier to predict the sentiment of the tweets correctly.
## Documentaion
Intially we remove all the neutral tweets from the database to ease the process.Then the tweets are fed into the preprocess function which cleans the tweets. The sentiment values are converted to numeric.A custom tokenizer is bulit to select the features for the machine learning algorithms.Tweets and sentiments are separated into training and test sets.Pipeline for the machine learning model is created and the training data is fitted to it.The accuracy on the test set comes **85%**.
**Due to lack of time, other machine learning models combined with other set of features couldn't be evaluated.**

