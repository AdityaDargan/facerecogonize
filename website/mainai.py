from flask import Flask,render_template, send_from_directory, url_for, request
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import cv2 
import os
 
import numpy as np
import face_recognition
import dlib
from datetime import datetime
from pymongo import MongoClient


################ MONGO DB #################
cluster0 = MongoClient("mongodb+srv://facerecognition:recognize423@cluster0.qnkcf4b.mongodb.net/?retryWrites=true&w=majority")
db=cluster0["data"]
collection=db["collection1"]
###########################################

count = collection.count_documents({})

app = Flask(__name__)

app.config['SECRET_KEY']='iamaditya'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
path = app.config['UPLOADED_PHOTOS_DEST']

photos = UploadSet('photos',IMAGES)
configure_uploads(app,photos)

def uploadimages(url,t2):
    global count
    if count==0:
        collection.insert_one({"_id":0,"count":0})
    else:
        mydict=collection.find({"_id":0}) 
        for m in mydict:
            count=m["count"]
    count+=1
    img = cv2.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    li = encode.tolist()

    collection.insert_one({'_id':count,"list":li,"name":t2})
    collection.update_one({"_id":0},{"$set":{"count":count}})
    return 


def get_data(classNames):  

    l2=[]
    cursor = collection.find({})
    for document in cursor:
        if document["_id"]!=0:
            l2.append(document["list"])
            classNames.append(document["name"])
    return l2

def MyRec(rgb,name,x,y,w,h,v=20,color=(200,0,0),thikness =2):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)

    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

def compare(img,bbox,width=180,height=227): # saving cropping images and comparing with already existing ones.

    classNames = []
    facelist = get_data(classNames)

    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height))
    #we need this line to reshape the images
    
    imgS = cv2.resize(imgCrop,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)


    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(facelist,encodeFace)
        faceDis = face_recognition.face_distance(facelist,encodeFace)

        matchIndex = np.argmin(faceDis)
        
        if faceDis[matchIndex]< 0.55:
            name = classNames[matchIndex].upper()
            # name = classNames[matchIndex]
        else: 
            name = 'Unknown'
        # print(faceDis[matchIndex])

        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        # cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        # cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        # print(name)
    return name

def facesearch(url): # identify all faces in the image 
    
    detector = dlib.get_frontal_face_detector()
    frame =cv2.imread(url)
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    names=[]

    for counter,face in enumerate(faces): # loop for going through all faces in image

        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        # cv2.rectangle(frame,(x1,y1),(x2,y2),(220,255,220),1)
        # MyRec(frame,str(counter), x1, y1, x2 - x1, y2 - y1, 10, (0,250,0), 3)
        names.append(compare(frame,(x1,y1,x2,y2)))
    
    return names
    # frame = cv2.resize(frame,(800,800))
    # cv2.imshow('img',frame)
    # cv2.waitKey(0)
    # print("done saving")






class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')
    

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'],filename)

@app.route('/upload',methods=['GET','POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        myList = os.listdir(path)

        for cl in myList:
            os.remove(f'{path}/{cl}')
        filename=photos.save(form.photo.data)
        file_url = url_for('get_file',filename=filename)
        url = file_url[1::]
        t2 = request.form['text']

        if not t2:
            error = 'Text field cannot be empty'
            return f'<script>alert("{error}");window.location.href="/upload";</script>'

        uploadimages(url,t2)

    else:
        t2=""
        file_url=None
    

    return render_template('upload.html',form=form,file_url=file_url,text=t2)

@app.route('/search',methods=['GET','POST'])
def search_image():
    form = UploadForm()
    if form.validate_on_submit():
        myList = os.listdir(path)
        for cl in myList:
            os.remove(f'{path}/{cl}')
        filename=photos.save(form.photo.data)
        file_url = url_for('get_file',filename=filename)
        url = file_url[1::]
        names = facesearch(url)
        t1=""
        for i in names:
            t1+=i
            t1+=" , "

    else:
        t1=""
        file_url=None
        
    return render_template('search.html',form=form,file_url=file_url,text = t1)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/search')
def about():
    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True)