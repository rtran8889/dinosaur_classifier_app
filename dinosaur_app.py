import streamlit as st 
import numpy as np
from fastai import * 
from fastai.vision.all import *
from fastai.vision.widgets import *
from PIL import Image
import base64

main_bg = 'imaginary.jpg'
main_bg_ext = 'jpg'

set_background = f"""
    <style>
    .stApp {{
        background-image: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        background-size: cover;
    }}
    </style>
    """

st.markdown(set_background, unsafe_allow_html=True)

st.title('Dinosaur Image Classifier')

with st.sidebar:
    st.write('''Welcome! This is an image classifier that can classify eleven different types of dinosaurs
    including the mighty Tyrannosaurus Rex, Brachiosaurus, Stegosaurus, Spinosaurus, Ankylosaurus, Triceratops, Velociraptor, Pteranodon, Parasaurolophus, Allosaurus, and the large Mosasaurus!''')
    st.subheader('Getting Started')
    st.write('1. Upload an image of a dinosaur into the file uploader on the right. ')
    st.write('NOTE: Your image should be from one of the eleven types of dinosaurs listed above.')
    st.write('2. Press the predict button and wait for the AI to classify your dino image!')
    st.write('3. Play around with it and have fun!')


    file = open('dino.gif', 'rb')
    contents = file.read()
    data_url = base64.b64encode(contents).decode('utf-8')
    file.close()
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" width="300">', unsafe_allow_html=True)


c1, c2 = st.beta_columns(2)


uploaded_file = c1.file_uploader("Insert An Image... ", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None: 
    try: 
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        c2.image(image, caption='Uploaded Dino.')
        c1.markdown('**Classifying...**')
        if c1.button('Predict'): 
            inf_learner = load_learner('dinosaur.pkl')
            pred, p, prob = inf_learner.predict(img_array)
            c2.markdown('**This is an image of a {}**'.format(pred))
    except: 
        c1.write('Image cannot be requested. Please upload another image')
