import pyrealsense2 as rs
import open3d as o3d
import numpy as np

# Configuração do pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Inicializa o pipeline
pipeline.start(config)

# Captura um único frame
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Converte as imagens para arrays numpy
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Converte as imagens para Open3D
depth_o3d = o3d.geometry.Image(depth_image)
color_o3d = o3d.geometry.Image(color_image)

# Gera a nuvem de pontos
rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(color_o3d, depth_o3d)
intrinsic = o3d.camera.PinholeCameraIntrinsic(960, 540, 525, 525, 480, 320)
pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)

# Visualiza a nuvem de pontos
o3d.visualization.draw_geometries([pcd])

pipeline.stop()
