import streamlit as st
import cv2
import numpy as np
from PIL import Image

import cv2
import numpy as np

def cartoonify_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Edge mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)

    # Color quantization
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    k = 9  # number of colors
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    # Combine edges and quantized image
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)

    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    return cartoon

def pencil_sketch(img):
    img = np.array(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_blur = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(img_gray, 255 - img_blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

def sepia_filter(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, sepia_matrix)
    sepia = np.clip(sepia, 0, 255)
    sepia = cv2.cvtColor(sepia.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return sepia

st.title("🎨 AI Art Filter App!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_container_width=True)

    filter_option = st.selectbox("Choose a filter", ["Cartoonify", "Pencil Sketch", "Sepia"])

    if st.button("Apply Filter"):
        if filter_option == "Cartoonify":
            output_img = cartoonify_image(img)
        elif filter_option == "Pencil Sketch":
            output_img = pencil_sketch(img)
        elif filter_option == "Sepia":
            output_img = sepia_filter(img)

        st.image(output_img, caption=f"{filter_option} Image", use_container_width=True)
