import os
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
import pandas as pd
import tweepy
import numpy as np
from tweepy import OAuthHandler
from textblob import TextBlob
from keras.preprocessing.text import Tokenizer
from termcolor import colored
from keras.preprocessing.sequence import pad_sequences
import pickle
import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)

consumer_key = 'QlEsya6A9B7s6M2WocOiljUut'
consumer_secret = 'JjkUDRIeb7k79PPPAsGHdjlCza0DLBjPTlgtj8gxoYLSd4TD4T'
access_token = '1207959500973625344-ZKqPDBf6AYUz4BBfaCaXBrzNFk1oGP'
access_token_secret = 'Z0ORih1UZUfeL2yhwUqTkHudroEFZbjMPPNKc0ERs0J0T'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

#fetching tweets
api = tweepy.API(auth)
texts = []
print("enter keyword")
q = input()
public_tweet = api.search(q,count=100,lang='en')
for t in public_tweet:
	texts.append(t.text)
d = {'Tweet':texts}
df = pd.DataFrame(d)
df.to_csv('demofile.csv')
print("file created")

import preprocessing

test_data = pd.read_csv('demo_clean_test.csv')
test_data["sentiment"]=2
test_data["value"]=5.0

#tokenizing
tokenizer = Tokenizer(num_words = 2000, split = ' ')
tokenizer.fit_on_texts(test_data['Clean_tweet'].astype(str).values)
test_tweets = tokenizer.texts_to_sequences(test_data['Clean_tweet'].astype(str).values)
max_len = max([len(i) for i in test_tweets])

test_tweets = pad_sequences(test_tweets, maxlen = 40)
testarray = np.array(test_tweets)


newmodel = pickle.load(open('lstmmodel.sav','rb'))
predictedval = newmodel.predict(testarray, batch_size=128)

print("doing analysis ...")
i=0
for p in predictedval:
	if float(p[0])>=0.45 and float(p[0])<=0.55:
		test_data["sentiment"][i] = 0
		test_data["value"][i] = float(p[0])
	elif float(p[0]) > 0.55:
		test_data["sentiment"][i] = 1
		test_data["value"][i] = float(p[0])
	else:
		test_data["sentiment"][i] = -1
		test_data["value"][i] = float(p[0])
	i+=1

test_data = test_data.drop('Clean_tweet',axis=1)
test_data.to_csv('demo_clean_test.csv',index=False)
print("Done :)")


dataset = pd.read_csv('demo_clean_test.csv',encoding='latin-1')
count = dataset['sentiment'].value_counts()

plt.figure(figsize = (12,8))
plt.xticks([-1,0,1],['negative','neutral','positive'])
plt.xticks([-1,0,1])

bar = plt.bar(x = count.keys(),height=count.values,color=['r','b','g'])
print(bar[0])
h=[]
for i in bar.patches:
    h.append(i.get_height())

bar[0].set_color('g')
bar[1].set_color('r')
bar[2].set_color('b')
plt.xlabel("Tweet sentiment")
plt.ylabel("Tweet count")
plt.title("Count of tweets for each sentiment")
xx = ''.join([q,'barchart.png'])
plt.savefig(xx)

#pie chart
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = ['positive', 'negative', 'neutral']
students = h
ax.pie(students, labels = langs,autopct='%1.2f%%')
xx = ''.join([q,'piechart.png'])
plt.savefig(xx)

positive_tweets = ' '.join(dataset['Tweet'].str.lower())

wordcloud = WordCloud(stopwords = STOPWORDS, background_color = "white", max_words = 1000).generate(positive_tweets)
plt.figure(figsize = (12, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Positive tweets Wordcloud")
xx = ''.join([q,'wordcloud.png'])
plt.savefig(xx)