import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load the pre-trained model and feature extractor
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

def generate_point_cloud(depth_output):
    # Convert depth output to 3D point cloud (X, Y, Z)
    h, w = depth_output.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_output

    # Stack points into (X, Y, Z) coordinates
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    
    return points

def plotly_point_cloud(points):
    # Convert the point cloud into a Plotly 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=points[:, 2], colorscale='Viridis', opacity=0.8)
    )])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    return fig

# Streamlit UI
st.title("3D Point Cloud and Mesh Generation from a Single Image")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_image:
    # Read the uploaded image
    image = np.array(Image.open(uploaded_image))
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Generate depth map and point cloud
    image_rgb, depth_output = process_image(image)
    
    # Display depth image
    st.subheader("Depth Image")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(depth_output, cmap='plasma')
    ax.axis('off')
    st.pyplot(fig)
    
    # Generate point cloud
    points = generate_point_cloud(depth_output)
    
    # Show point cloud using Plotly
    st.subheader("3D Point Cloud")
    st.write("Here is the 3D point cloud generated from the depth map.")
    point_cloud_fig = plotly_point_cloud(points)
    st.plotly_chart(point_cloud_fig)
    
    # Optional: Generate mesh if required using Open3D
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_filtered, alpha=0.03)
    # mesh_fig = plotly_point_cloud(mesh) # If you need to plot the mesh as well
    # st.plotly_chart(mesh_fig)
