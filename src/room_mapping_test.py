import pyrealsense2 as rs
import numpy as np
import open3d as o3d

def capture_point_cloud(pipeline, intrinsics, depth_scale):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None

    depth_image = np.asanyarray(depth_frame.get_data())
    o3d_depth = o3d.geometry.Image(depth_image)
    
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth, 
        o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, 
            intrinsics.height, 
            intrinsics.fx, 
            intrinsics.fy, 
            intrinsics.ppx, 
            intrinsics.ppy
        ),
        depth_scale=1.0/depth_scale, 
        depth_trunc=3.0
    )
    return pcd

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    video_stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = video_stream_profile.get_intrinsics()

    print("Capturando uma nuvem de pontos...")
    pcd = capture_point_cloud(pipeline, intrinsics, depth_scale)
    if pcd is None:
        print("Falha na captura do frame. Tentando novamente...")
    else:
        print("Visualizando a nuvem de pontos...")
        o3d.visualization.draw_geometries([pcd])

    pipeline.stop()

if __name__ == '__main__':
    main()
