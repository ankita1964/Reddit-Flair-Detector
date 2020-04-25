from flask import Flask, render_template, url_for, request, jsonify, send_file
import praw
import pickle
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
import json
from json import JSONEncoder

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
    
def detect_flair_txt(post_url):
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
    return str(output)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

@app.route('/automated_testing', methods=['POST'])
def automated_testing():
    #f = open("upload_file/file.txt", "rb")
    f = request.files['upload_file']
    lines = [x.decode('utf8').strip() for x in f.readlines()]
    inp = []
    out = []
    output_dict = {}
    for line in lines:
        print(line)
        print(type(line))
        inp.append(line)
        output = detect_flair_txt(line)
        out.append(output)
        print(output)
        print(type(output))
    #for i in out: 
    #    i = i.decode('ascii')
    
    for key in inp: 
        for value in out: 
            output_dict[key] = value 
            out.remove(value) 
            break  
    print(output_dict)
    json_obj = json.dumps(output_dict, indent = 3)
    print(json_obj)
    with open("result.json", "w") as output_file:
        output_file.write(json_obj)
    return send_file("result.json")
if __name__ == "__main__":
    app.run(debug=True)