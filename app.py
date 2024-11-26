import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import open3d as o3d
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

def generate_point_cloud(image_rgb, depth_output):
    # Convert OpenCV image to PIL Image
    image_pil = Image.fromarray(image_rgb)
    image_cropped = image_pil.crop((16, 16, image_pil.width - 16, image_pil.height - 16))
    
    # Prepare depth image for open3d
    depth_image = (depth_output * 255 / np.max(depth_output)).astype('uint8')
    image = np.array(image_cropped)
    
    # Create Open3D RGBD image
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)
    
    # Create camera intrinsics
    width, height = image_cropped.size
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)
    
    # Create point cloud
    pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    
    return pcd_raw

def remove_outliers(pcd_raw):
    # Remove statistical outliers from point cloud
    cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
    pcd = pcd_raw.select_by_index(ind)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    
    return pcd

def create_mesh_from_point_cloud(pcd):
    # Surface reconstruction to create a mesh from point cloud
    print("Creating mesh from point cloud...")
    # Poisson Surface Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    return mesh

def plotly_point_cloud(pcd):
    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Plotting point cloud using Plotly
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

def plotly_mesh(mesh):
    # Convert Open3D mesh to numpy arrays
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Plotting mesh using Plotly
    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightgray',
        opacity=0.5
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
    pcd_raw = generate_point_cloud(image_rgb, depth_output)
    
    # Show point cloud without outlier removal using Plotly
    st.subheader("3D Point Cloud (Raw)")
    st.write("Here is the raw point cloud generated from the depth map.")
    point_cloud_fig = plotly_point_cloud(pcd_raw)
    st.plotly_chart(point_cloud_fig)
    
    # Show point cloud with outlier removal using Plotly
    st.subheader("3D Point Cloud with Outlier Removal")
    pcd_filtered = remove_outliers(pcd_raw)
    st.write("Here is the point cloud after removing statistical outliers.")
    point_cloud_filtered_fig = plotly_point_cloud(pcd_filtered)
    st.plotly_chart(point_cloud_filtered_fig)
    
    # Create mesh from point cloud
    st.subheader("Mesh from Point Cloud")
    mesh = create_mesh_from_point_cloud(pcd_filtered)
    st.write("Here is the mesh generated from the point cloud.")
    
    # Display the mesh using Plotly
    mesh_fig = plotly_mesh(mesh)
    st.plotly_chart(mesh_fig)
    
    # Optionally, save the mesh
    mesh_filename = "output_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_filename, mesh)
    st.download_button(
        label="Download Mesh (.ply)",
        data=open(mesh_filename, "rb").read(),
        file_name=mesh_filename,
        mime="application/octet-stream"
    )
