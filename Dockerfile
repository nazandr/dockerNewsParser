FROM python:3-onbuild

RUN apt-get -yqq update
RUN pip install -r requirements.txt
RUN easy_install -U gensim

CMD KERAS_BACKEND=theano python ./bot.py