from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# import tensorflow as tf
# from keras.models import load_model
# from keras.preprocessing import image
# import json
# from tensorflow import Graph, Session

# img_height,img_width=224,224
# with open('./models/model_adam.json','r') as f:
#     labelInfo=f.read()

# labelInfo=json.loads(labelInfo) 

# model_graph= Graph()
# with model_graph.as_default():
#     tf_session= Session()
#     with tf_session.as_default():
#         model=load_model('./models/model_100_eopchs_adam_20190807.h5')



# # Create your views here.
# def index(request):
#     context={'a':1}
#     return render(request,'index1.html',context)


# def classifier(request):
#     fileObj=request.FILES['fileP']
#     fs=FileSystemStorage()
#     path=fs.save(fileObj.name,fileObj)
#     path=fs.url(path)
#     testimage='.'+path

#     img=image.load_img(testimage,target_size=(img_height,img_width))
#     x=image.img_to_array(img)
#     x=x/255 
#     x=x.reshape(1,img_height,img_width,3)

#     with model_graph.as_default():
#         with tf_session.as_default():
#             pred=model.predict(x)


#     import numpy as np
#     prediction=labelInfo[str(np.argmax(pred[0]))]



    

#     context={'path':path,
#     'prediction':prediction[1],
#     }
#     return render(request,'index.html',context)

# def viewhistory(request):
#     import os
#     History=os.listdir('./media/')
#     historypath=['./media/'+i for i in History]
#     context={
#         'historypath':historypath
#     }

#     return render(request,'history.html',context)
    
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

import numpy as np
import h5py
from PIL import Image
import PIL
import os



MODEL_ARCHITECTURE = './model/model_adam.json'
MODEL_WEIGHTS = './model/model_100_eopchs_adam_20190807.h5'
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')

# ::: MODEL FUNCTIONS :::
def model_predict(filepath, model):
	

	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	IMG = image.load_img(filepath).convert('L')
	print(type(IMG))

	# Pre-processing the image
	IMG_ = IMG.resize((257, 342))
	print(type(IMG_))
	IMG_ = np.asarray(IMG_)
	print(IMG_.shape)
	IMG_ = np.true_divide(IMG_, 255)
	IMG_ = IMG_.reshape(1, 342, 257, 1)
	print(type(IMG_), IMG_.shape)

	print(model)

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
	prediction = model.predict_classes(IMG_)

	return prediction



def index(request):
    context={'a':1}
    return render(request,'index1.html',context)


def classifier(request):

	
    classes = {'TRAIN': ['BACTERIA', 'NORMAL', 'VIRUS'],
	           'VALIDATION': ['BACTERIA', 'NORMAL'],
	           'TEST': ['BACTERIA', 'NORMAL', 'VIRUS']}
    fileObj=request.FILES['fileP']
    fs=FileSystemStorage()
    path=fs.save(fileObj.name,fileObj)
    path=fs.url(path)
	
    filepath='.'+path
	

	

    prediction = model_predict(filepath, model)
    context={'path':path,
    'prediction':str(classes['TRAIN'][prediction[1]])}
    return render(request,'index.html',context)


# def viewhistory(request):
#     import os
#     History=os.listdir('./media/')
#     historypath=['./media/'+i for i in History]
#     context={
#         'historypath':historypath
#     }

#     return render(request,'history.html',context)
