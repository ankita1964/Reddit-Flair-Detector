from flask import Flask, render_template, url_for, request
import praw
import pickle
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

lr_model = pickle.load(open('lr_model.pkl', 'rb'))

with open("reddit_secret_keys.json") as f:
    param = json.load(f)

@app.route('/')
def home():
    return render_template("index.html")

replace_by_space = re.compile('[/(){}\[\]\|@,;]')
bad_symbols = re.compile('[^0-9a-z #+_]')
stop_words = stopwords.words('english')

def clean_data(text):
    #converting to lowercase
    text = text.lower()
    #re.sub(new_value, text_to_processed) 
    text = replace_by_space.sub(' ', text)
    text = bad_symbols.sub('', text)
    #removing the stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words) 
    return text

def clean_url(u):
    if u.startswith("http://"):
        u = u[7:]
    if u.startswith("https://"):
        u = u[8:]
    if u.startswith("www."):
        u = u[4:]
    if u.endswith("/"):
        u = u[:-1]
    return u

def reddit_url(u):
    u = u.replace('redditcom', '')
    u = u.replace('r', '')
    u = u.replace('india', '')
    u = u.replace('comments','')
    for word in u:
        u = ' '.join(u.split('_'))
    return u

@app.route('/detect_flair', methods=['POST'])
def detect_flair():
    post_url = request.form['post_url']
    post_url = post_url.lower()
    reddit = praw.Reddit(client_id = param['client_id'],
                     client_secret = param['api_key'],
                     user_agent = param['useragent'])
    submission = reddit.submission(url=post_url)
    post_dict = {}
    post_dict['title'] = submission.title
    post_dict['body'] = submission.selftext
    post_dict['url'] = submission.url
    submission.comments.replace_more(limit=None)
    comment = ''
    for top_level_comment in submission.comments:
        comment = comment + ' ' + top_level_comment.body
    post_dict["comments"] = comment

    post_dict['title'] = clean_data(post_dict['title'])
    post_dict['comments'] = clean_data(post_dict['comments'])
    post_dict['body'] = clean_data(post_dict['body'])

    post_dict['url'] = clean_url(post_dict['url'])
    post_dict['url'] = clean_data(post_dict['url'])
    post_dict['url'] = reddit_url(post_dict['url'])
    post_dict['url'] = clean_data(post_dict['url'])
    post_dict['title_comments_body_url'] = post_dict['title'] + ' ' + post_dict['comments'] + ' ' + post_dict['body'] + ' ' + post_dict['url']
    output = lr_model.predict([post_dict['title_comments_body_url']])
    return render_template('index.html', detected_flair = 'The flair for the post is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)