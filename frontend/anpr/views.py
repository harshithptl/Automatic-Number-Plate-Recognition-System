from django.shortcuts import render
from PIL import Image
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
from anpr.pro import extraction
from anpr.ocr import character
# Create your views here.

def home(request):
    path=os.getcwd()
    dir=path+'\media'
    
    img={}
    if request.method=='POST':
        file=request.FILES['submit-btn']
        fs= FileSystemStorage()
        name=fs.save(file.name, file)
        img_path=dir+'\\'+name
        print(name)
        process=extraction(img_path,name,0.5)
        licence_plate=process.read_img()
        img['url']=fs.url(name)
        img['ocr_processed']=character(licence_plate)
        print(img)
    return render(request, "home.html",img)
