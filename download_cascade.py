import urllib.request
import os

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
filename = "haarcascade_frontalface_default.xml"

print("Downloading Haar cascade...")
urllib.request.urlretrieve(url, filename)
print("Download completed!")