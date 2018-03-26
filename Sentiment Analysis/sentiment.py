import pandas as pd 
import numpy as np
import string, re
import nltk


stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
punctuation = list(string.punctuation)
stop_list = stop_list + punctuation +["rt", 'url']

data = pd.read_csv('Sentiment.csv')

#Deleting all neutral sentiments
for i in range(len(data.sentiment)):
    if(data.sentiment[i])=='Neutral':
        data=data.drop(i)
    
def preprocess(tweet):
    if type(tweet)!=type(2.0):
        tweet = tweet.lower()
        tweet = " ".join(tweet.split('#'))
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))','URL',tweet)
        tweet = re.sub("http\S+", "URL", tweet)
        tweet = re.sub("https\S+", "URL", tweet)
        tweet = re.sub('@[^\s]+','AT_USER',tweet)
        tweet = tweet.replace("AT_USER","")
        tweet = tweet.replace("URL","")
        tweet = tweet.replace(".","")
        tweet = tweet.replace('\"',"")
        tweet = tweet.replace('&amp',"")
        tweet  = " ".join([word for word in tweet.split(" ") if word not in stop_list])
        tweet  = " ".join([word for word in tweet.split(" ") if re.search('^[a-z]+$', word)])
        tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split(" ")])
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = tweet.strip('\'"')
    else:
        tweet=''
    return tweet

data['processed_text'] = data.text.apply(preprocess)
categories = data.sentiment.unique()
categories  = categories.tolist()

tweets = list(data.processed_text.values)
sentiment = data.sentiment.values.tolist()

for sent in range(len(sentiment)):
    if(sentiment[sent]=='Positive'):
        sentiment[sent]=1
    elif(sentiment[sent]=='Negative'):
        sentiment[sent]=0

from nltk.stem import PorterStemmer
ps = PorterStemmer()
def custom_tokenizer(text, use_stem=True, stemmer=ps, use_pos=False, 
                     use_only_adj=False, use_bigrams=False, use_bigrams_only=False):
    # Separate words
    words = word_tokenize(text)
    # PoS tagging words
    if use_pos:
        pos_tags = nltk.pos_tag(words)
    else:
        pos_tags = zip(words, [''] * len(words))
    
    tokens = []
    # Special treatment for bigrams
    if use_bigrams:
        tokens += list(ngrams(words, n=2))
        if use_bigrams_only:
            return tokens
        else:
            tokens += [(x,) for x in words]
        return tokens
    
    for word, tag in pos_tags:
        res_word = word
        use_word = True
        # Convert to stem
        if use_stem:
            res_word = stemmer.stem(res_word)
        # Use POS tag with the word
        if use_pos and not use_only_adj:
            res_word += '_' + tag
        # Only use adv and adj
        if use_only_adj and not (tag[:2] == 'JJ' or tag[:2] == 'RB'):
            use_word = False
        # Append the word to the tokenizer
        if use_word:
            tokens.append(res_word)
    return tokens

def text_stems_tok(text):
    return custom_tokenizer()
def pos_tok(text):
    return custom_tokenizer(text, use_stem=False, use_pos=True)
def pos_stems_tok(text):
    return custom_tokenizer(text, use_stem=True, use_pos=True)
def adj_tok(text):
    return custom_tokenizer(text, use_stem=False, use_pos=True, use_only_adj=True)
def adj_stems_tok(text):
    return custom_tokenizer(text, use_stem=True, use_pos=True, use_only_adj=True)
def unigrams(text):
    return word_tokenize(text)
def uni_bigrams(text):
    return custom_tokenizer(text, use_bigrams=True)
def bigrams(text):
    return custom_tokenizer(text, use_bigrams=True, use_bigrams_only=True)
def uni_bigrams_stems(text):
    tokens = uni_bigrams(text)
    res_tokens = []
    for t in tokens:
        res_tokens.append(tuple([ps.stem(x) for x in t]))
    return res_tokens

X=tweets
y=sentiment
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer(tokenizer=uni_bigrams_stems)),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier())
                    ])
    
text_clf = text_clf.fit(X_train,y_train)

predicted = text_clf.predict(X_test)
np.mean(predicted == y_test)

