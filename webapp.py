import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
#import tensorflow as tf
from re import DEBUG,sub
from flask import Flask,render_template,request,redirect,send_file,url_for,Response
from werkzeug.utils import secure_filename,send_from_directory
import os
#from popen import subprocess
import re
import requests
import shutil
import time
import glob
from ultralytics import YOLO
app=Flask(__name__)
upload_folder=os.path.join('static','uploads')
app.config['UPLOAD_FOLDER']=upload_folder
@app.route('/video')
def index():
    return render_template("Home page.html")

@app.route('/sign-up')
def page1():
    return render_template("page 1.html")

@app.route("/image",methods=["GET","POST"])
def predict_img():
    if request.method=="POST":
        f=request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
        model = YOLO('yolov8n.pt')  # load an official model
        img="C:\\Users\\Gowthami\\PycharmProjects\\human_detection\\static\\uploads\\"+secure_filename(f.filename)
        results = model.predict(img,show=True,save=True)
        folder_path='runs/detect'
        sub_folders=[f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path,f))]
        latest_subfolder=max(sub_folders,key=lambda x:os.path.getctime(os.path.join(folder_path,x)))
        directory=folder_path+'/'+latest_subfolder
        print("printing directory:",directory)
        files=os.listdir(directory)
        latest_file=files[0]
        print(latest_file)
        filename=os.path.join(folder_path,latest_subfolder,latest_file)
        environ=request.environ
        return send_from_directory(directory,latest_file,environ)
    return render_template('upload_new.html')

@app.route("/",methods=["GET",'POST'])
def predict_video():
    if request.method=="POST":
        f=request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
        model=YOLO('yolov8n.pt')
        video_path="C:\\Users\\Gowthami\\PycharmProjects\\human_detection\\static\\uploads\\"+secure_filename(f.filename)
        cap=cv2.VideoCapture(video_path)
        frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter('output.mp4',fourcc,30.0,(frame_width,frame_height))
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:
                break
            results=model(frame,save=True)
            print(results)
            cv2.waitKey(1)
            res_plotted=results[0].plot()
            cv2.imshow("result",res_plotted)
            out.write(res_plotted)
            if cv2.waitKey(1)==ord('q'):
                break
        return video_feed()
    return render_template('upload_video.html')

def get_frame():
    folder_path=os.getcwd()
    mp4_files='output.mp4'
    video=cv2.VideoCapture(mp4_files)
    while True:
        success,image=video.read()
        if not success:
            break
        ret,jpeg=cv2.imencode('.jpg',image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+jpeg.tobytes()+b'\r\n\r\n')
        time.sleep(0.1)

@app.route("/")
def video_feed():
    return Response(get_frame(),mimetype='multipart/x-mixed-replace;boundary=frame')




if __name__=='__main__':
    app.run(debug=True)