import requests

files = {'upload_file': open('file.txt','rb')}
r = requests.post('http://127.0.0.1:5000/automated_testing', files=files)
