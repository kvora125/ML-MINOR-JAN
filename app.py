import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
import os
import matplotlib as plt
model=pickle.load(open('img_model.p','rb'))
st.title('Image Classification System using Machine Learning')
st.text('upload a image')
uploaded_file=st.file_uploader("choos an image....", type="jpg")
x=False;
if uploaded_file is not None:
  img=Image.open(uploaded_file)
  st.image(img,caption='uploaded image')
  x=True;
CATEGORIES = ['apple','banana','mango','orange']
if st.button('PREDICT'):
  if x:
    flat_data=[]
    img=np.array(img)
    img_resized = resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.write('result...')
    # st.write(f' PREDICTED OUTPUT: {y_out}')
    q=model.predict_proba(flat_data)
    for index,item in enumerate(CATEGORIES):
      st.write(f'{item} : {q[0][index]*100}%')