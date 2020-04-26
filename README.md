# Reddit-Flair-Detector
Reddit Flair Detector for the subreddit 'India' 

<h3> About: </h3>
The Reddit Flair Detector is an application used to detect the flair of a reddit post using the link to the reddit post. The application can be found <a href="https://secure-sands-63470.herokuapp.com/">here</a>.
An endpoint to the application is present at: https://secure-sands-63470.herokuapp.com/automated_testing. An automated post request send to this link will return a json file.

The request can be sent using: 
~~~
import requests

files = {'upload_file': open('file.txt','rb')}
r = requests.post('https://secure-sands-63470.herokuapp.com/automated_testing', files=files)
~~~

1. <a href= "https://github.com/gaurigupta31/Reddit-Flair-Detector/blob/master/Reddit-Data-Collection.ipynb">Reddit Data Collection</a>
2. <a href= "https://github.com/gaurigupta31/Reddit-Flair-Detector/blob/master/Exploratory-Data-Analysis.ipynb">Exploratory Data Analysis </a>
3. <a href = "https://github.com/gaurigupta31/Reddit-Flair-Detector/blob/master/Reddit-Flair-Detector.ipynb">Building Flair Detector </a>
4. <a href="https://github.com/gaurigupta31/Reddit-Flair-Detector/tree/master/app">Web App </a>
5. <a href="https://secure-sands-63470.herokuapp.com/">Deployment </a>

<h3> Development Environment: </h3>

1. Download/Clone the repo
2. Run: 
~~~ 
pip install -r requirements.txt
~~~
3. To run the application locally: 
~~~
cd folder-name
py app.py
~~~


<h3> About the Detector: </h3> 
The detector uses scikit's Logistic Regression model for classification of the posts for different flairs of the r/India subreddit. The model uses the posts' title, body (or self-text), comments and url as a combined feature. The accuracy achieved with imputed null values and combined feature is 78-81%. 
