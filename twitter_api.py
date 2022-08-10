import re
import configparser
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sentiment_analysis_spanish import sentiment_analysis

# #leyendo config
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#autenticacion de cuenta en api de twitter en
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
# for status in tweepy.Cursor(api.search_tweets, "bogota",geocode='4.624335,-74.063644,30km', lang = 'es',
#     count=18000).items(500):
#     print(status.text)

public = tweepy.Cursor(api.search_tweets, "bogota",geocode='4.624335,-74.063644,40km', lang = 'es',
    count=18000).items(2000)

# #api.search_tweets(q = '', count = 15000, geocode='4.624335,-74.063644,30km', lang = 'es', locale = 'es', result_type = 'mixed', until = '2022-07-23')

sentimiento = sentiment_analysis.SentimentAnalysisSpanish()

def process_twitter_features(message):

    message = re.sub(r'[\.\,]http','. http', message)
    message = re.sub(r'[\.\,]#', '. #', message)
    message = re.sub(r'[\.\,]@', '. @', message)

    # eliminar menciones, hashtags y URL
    message = re.sub(r'((?<=\s)|(?<=\A))(@|#)\S+', '', message)
    message = re.sub(r'\b(https?:\S+)\b', '', message)

    return message

col = ['numTweet','tweet', 'analisis', "rango"]
data = []
aux = 0
for tweet in public:
    aux = aux + 1
    tweetNormalizado = process_twitter_features(tweet.text)
    if sentimiento.sentiment(tweetNormalizado)<0.4 :
        data.append([aux, tweetNormalizado, sentimiento.sentiment(tweetNormalizado), 0])
    elif sentimiento.sentiment(tweetNormalizado)>0.6 :
        data.append([aux, tweetNormalizado, sentimiento.sentiment(tweetNormalizado), 1])
    else :
        data.append([aux, tweetNormalizado, sentimiento.sentiment(tweetNormalizado), 0.5])

    # print(process_twitter_features(tweet.text))
    # print(sentimiento.sentiment(tweet.text))
    # print((tweet.text))

# print(data)


df = pd.DataFrame(data, columns=col)

# print(df)

df.to_csv("tweets-resultados.csv")

sb.scatterplot(data = df, x="analisis", y="numTweet", hue="rango")

plt.show()


X = "Esta muy buena esa pelicula"
print(X,": ",sentimiento.sentiment(X))

Y= "Nunca jugaron juntos en la infancia"
print(Y,": ",sentimiento.sentiment(Y))

