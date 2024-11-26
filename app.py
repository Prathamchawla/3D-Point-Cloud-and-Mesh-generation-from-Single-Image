import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import cv2
from PIL import Image

# Load pre-trained model
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

def process_image(image):
    # Convert the image to PIL format before passing to feature extractor
    image_pil = Image.fromarray(image)  # Convert NumPy array to PIL image
    
    # Preprocess image
    image_resized = image_pil.resize((640, 480))  # Resize image to match model input
    
    image_rgb = np.array(image_resized)  # Convert back to NumPy array if necessary
    
    # Get depth predictions
    inputs = feature_extractor(images=image_resized, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Post processing
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]

    return image_rgb, output

def generate_point_cloud(image_rgb, depth_output):
    # Convert image to 3D point cloud (X, Y, Z) coordinates
    height, width, _ = image_rgb.shape
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    X, Y = np.meshgrid(x, y)
    Z = depth_output

    # Flatten the arrays
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    return X, Y, Z

# Streamlit UI
st.title("3D Point Cloud Generation from a Single Image")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_image:
    # Read the uploaded image
    image = np.array(Image.open(uploaded_image))
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Generate depth map and point cloud
    image_rgb, depth_output = process_image(image)
    
    # Generate point cloud
    X, Y, Z = generate_point_cloud(image_rgb, depth_output)
    
    # Plot 3D point cloud using Plotly
    st.subheader("3D Point Cloud")
    trace = go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(size=2, color=Z, colorscale='Viridis', opacity=0.8)
    )
    layout = go.Layout(
        title="3D Point Cloud",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)
