import pandas as pd
import numpy as np 
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image



X,y =fetch_openml("mnist_784",version=1,return_X_y)

X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=0,train_size=7500,test_size=2500)


X_train_scaled= X_train/255
X_test_scaled=X_test/255

classifier=LogisticRegression(solver="saga",multi_class="multinomial").fit(X_test_scaled,y_train)

def get_prediction():
    im_pil =Image.Open(image)
    image_bw= im_pil.convert("L")
    image_bw_resize=image_bw.resize((28,28),Image.ANTALAIS)
    pixel_filter= 20
    min_pixel= np.percentile(image_bw_resize,pixel_filter)
    image_bw_resize_inverted_scale= np.clip(image_bw_resize-min_pixel,0,255)
    max_pixel= np.max(image_bw_resize)
    image_bw_resize_inverted_scale=np.asarray(image_bw_resize_inverted_scale)/max_pixel
    test_sample= np.array(image_bw_resize_inverted_scale).reshape(1,784)
    test_prediction= classifier.predict(test_sample)
    return test_prediction[0]
    
    