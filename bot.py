import telebot 
from telebot import types
import gensim
from keras.preprocessing import text
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import feedparser
import numpy as np
import time
from config import *

bot = telebot.TeleBot(token)

w2v = gensim.models.Word2Vec.load_word2vec_format('./news.model.bin', binary=True)
n_vocab = len(w2v['дом'])

def sendMessage(item, link):
	keyboard = types.InlineKeyboardMarkup()
	url = 'https://getpocket.com/edit?url=' + link
	url_button = types.InlineKeyboardButton(text="Добавить в Pocket", url=url)
	keyboard.add(url_button)
	bot.send_message(channel, item, reply_markup=keyboard)
	time.sleep(1)

def phrase2vec(phrase):
    vec = []
    phrase = text.text_to_word_sequence(phrase.lower())
    for word in phrase:
        if len(vec) == 0:
            vec = np.array([0] * n_vocab)
        try:
            seq = w2v[word]
            vec = vec + seq
        except KeyError:
            vec = vec + 0
    return vec

def newsParser(link):
	global lastNews
	feed = feedparser.parse(link)
	if feed.entries[0].title != lastNews:
		lastNews = feed.entries[0].title
		pred = phrase2vec(lastNews)
		pred = np.reshape(pred, (1,n_vocab))
		pred = model.predict_on_batch(pred)
		if pred[0,0] > 0.4:
			item = str(pred[0,0]) + ' ' + feed.entries[0].link
			print(feed.entries[0].title)
			sendMessage(item, feed.entries[0].link)

print('Length of vocabulary: ', n_vocab)

print('Model compiling...')
model = Sequential()
model.add(Dense(400, input_dim=n_vocab, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.load_weights('./weights1.h5')
model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

lastNews = ''

print('Start predicting...')

while True:
	newsParser('https://habrahabr.ru/rss')
	time.sleep(60 * 4)