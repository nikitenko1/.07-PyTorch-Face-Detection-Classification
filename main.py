import streamlit as st
from utils import set_background
from PIL import Image
import torch
from classifier import classifier
from io import BytesIO
import base64
from model import CNN

st.set_page_config(
    page_title='Football Players Classification',
    layout='centered'
)

set_background('utils/br_2.jpeg')

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        color: #000000;
        font-weight: bold;        
        margin-top: -75px;
    }
    .header {
        display: flex;
        justify-content: center;  /* Center horizontally */
        align-items: center;  /* Center vertically (if needed) */
        text-align: center;  
        font-size: 30px;
        color: #000000;
        white-space: nowrap;
        margin-top: -20px;
        width: 100%;  /* Ensures full width */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Football Players Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Upload an image to classify it as Thibaut Courtois, Mohamed Salah, Paulo Dybala  and Toni Kroos.</div>', unsafe_allow_html=True)

file = st.file_uploader('',type = ['jpg','jpeg','png','jfif'])

model = CNN()
model.load_state_dict(torch.load('checkpoints/best_model.pth', weights_only=True))
model.eval()  # Set to evaluation mode

class_names = {0:'Thibaut Courtois', 1:'Mohamed Salah', 2:'Paulo Dybala', 3:'Toni Kroos'}

if file is not None:
    
    image = Image.open(file).convert('RGB')

    prediction, score = classifier(image, model, class_names)
    
    if score == 'Error':
        # st.error(prediction)
        st.markdown(
                f"""
                <div style="color: black; font-size: 18px; font-weight: bold; background-color: #BFFF00; padding: 10px; border-radius: 5px; text-align: center; text-align: left;">
                    <p style="color: red;"> {prediction} </p>
                    <p><strong>Image should be:</strong></p>
                    <p style="color:green">1. Having only one face</p>
                    <p style="color:green">2. Face should be clear</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.stop()
    
    bufferd = BytesIO()
    image.save(bufferd, format='PNG')
    img_base64 = base64.b64encode(bufferd.getvalue()).decode()

    # Display classification results with reduced gap and no extra space
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center;">        
        <img src="data:image/png;base64,{img_base64}" style="width: 400px; height: 370px; object-fit: cover; margin-top: -20px;"/>
        <div style="font-size:40px; font-weight:bold; margin-left: 20px; color: #BFFF00; white-space: nowrap;">
            <p> <strong>Result: {prediction}</strong></p>
            <p style="margin-top:-10px;"> <strong> Score: {score}% </strong> </p>
        </div>        
        </div>
        """,
        unsafe_allow_html=True
    )