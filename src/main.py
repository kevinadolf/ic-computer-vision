import math
import copy
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
from ultralytics import YOLO


class RealSenseCamera:
    """Interface com a Intel RealSense D456."""

    def __init__(self, width=1280, height=720, fps=30):
        print("inicializando a camera realsense...")
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        self.align = rs.align(rs.stream.color)
        self.intrinsics = self._get_o3d_intrinsics()
        print("camera realsense inicializada com sucesso.")

    def _get_o3d_intrinsics(self):
        video_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = video_profile.get_intrinsics()
        return o3d.camera.PinholeCameraIntrinsic(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.cx,
            cy=intr.cy,
        )

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned = self.align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                print("aviso: quadro de profundidade ou cor nao encontrado.")
                return None, None

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            return depth_image, color_image
        except RuntimeError as exc:
            print(f"erro ao obter quadros: {exc}")
            return None, None

    def stop(self):
        print("parando o pipeline da camera.")
        self.pipeline.stop()


class ObjectDetector:
    """Deteccao 2D baseada em YOLOv8."""

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        print(f"carregando o modelo yolo: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print("modelo yolo carregado com sucesso.")

    def detect(self, image):
        results = self.model(image, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                if box.conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()
                class_id = int(box.cls)
                detections.append(
                    {
                        "class_name": self.model.names[class_id],
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(box.conf),
                    }
                )
        return detections


class ShapeAnalyzer:
    """Analisa a nuvem segmentada e classifica em primitivas geometricas."""

    def __init__(self, sphere_tol=0.12, base_tol=0.18, diff_tol=0.25):
        self.sphere_tol = sphere_tol
        self.base_tol = base_tol
        self.diff_tol = diff_tol

    def analyze(self, object_pcd):
        if len(object_pcd.points) == 0:
            return None

        obb = object_pcd.get_oriented_bounding_box()
        extent = np.asarray(obb.extent, dtype=float)

        if np.any(extent <= 0):
            return None

        shape = self._classify(extent)
        pose = np.eye(4)
        pose[:3, :3] = obb.R
        pose[:3, 3] = obb.center

        return {
            "shape": shape,
            "pose": pose,
            "dimensions": extent,
            "obb": obb,
        }

    def _classify(self, dims):
        dims = np.asarray(dims, dtype=float)
        mean_dim = dims.mean()
        rel_diff = np.abs(dims - mean_dim) / max(mean_dim, 1e-6)

        if np.all(rel_diff < self.sphere_tol):
            return "sphere"

        idx_sorted = np.argsort(dims)
        small1, small2, large = dims[idx_sorted]
        base_similarity = abs(small1 - small2) / max(small2, 1e-6)
        height_difference = abs(large - ((small1 + small2) * 0.5)) / max(large, 1e-6)

        if base_similarity < self.base_tol and height_difference > self.diff_tol:
            return "cylinder"

        distinct_diffs = np.abs(np.diff(np.sort(dims))) / np.maximum(np.sort(dims)[1:], 1e-6)
        if np.all(distinct_diffs > self.base_tol):
            return "box"

        return "block"


class PoseEstimationPipeline:
    """Pipeline coarse-to-fine convertido para abordagem model-free."""

    def __init__(self, voxel_size=0.01, min_points=300, max_depth=5.0):
        self.camera = RealSenseCamera()
        self.detector = ObjectDetector()
        self.shape_analyzer = ShapeAnalyzer()
        self.voxel_size = voxel_size
        self.min_points = min_points
        self.max_depth = max_depth
        self.intrinsics = self.camera.intrinsics

    def run_single_frame(self):
        depth_image, color_image = self.camera.get_frames()
        if depth_image is None or color_image is None:
            return [], None

        scene_pcd = self._create_point_cloud(depth_image, color_image)
        detections = self.detector.detect(color_image)

        objects = []
        for det in detections:
            object_pcd = self._segment_object(scene_pcd, det["bbox"])
            if object_pcd is None or len(object_pcd.points) < self.min_points:
                continue

            denoised = object_pcd.voxel_down_sample(self.voxel_size)
            if len(denoised.points) >= self.min_points // 2:
                object_pcd = denoised

            analysis = self.shape_analyzer.analyze(object_pcd)
            if analysis is None:
                continue

            color = self._average_color(object_pcd)
            objects.append(
                {
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "shape": analysis["shape"],
                    "pose": analysis["pose"],
                    "dimensions": analysis["dimensions"],
                    "color": color,
                    "point_cloud": object_pcd,
                }
            )
        return objects, scene_pcd

    def _create_point_cloud(self, depth_image, color_image):
        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.camera.depth_scale,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsics)
        pcd.transform(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        return pcd

    def _segment_object(self, scene_pcd, bbox):
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(x1, 0, self.camera.width - 1))
        y1 = int(np.clip(y1, 0, self.camera.height - 1))
        x2 = int(np.clip(x2, 0, self.camera.width - 1))
        y2 = int(np.clip(y2, 0, self.camera.height - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        polygon = np.array(
            [
                [x1, y1, 0.0],
                [x2, y1, 0.0],
                [x2, y2, 0.0],
                [x1, y2, 0.0],
            ],
            dtype=np.float64,
        )

        volume = o3d.visualization.SelectionPolygonVolume()
        volume.orthogonal_axis = "Z"
        volume.axis_max = self.max_depth
        volume.axis_min = 0.05
        volume.bounding_polygon = o3d.utility.Vector3dVector(polygon)

        cropped = volume.crop_point_cloud(scene_pcd)
        return cropped

    def _average_color(self, object_pcd):
        colors = np.asarray(object_pcd.colors)
        if colors.size == 0:
            return np.array([0.8, 0.8, 0.8])
        return np.clip(colors.mean(axis=0), 0.0, 1.0)

    def stop(self):
        self.camera.stop()


class DigitalTwinVisualizer:
    """Renderizacao nao bloqueante do gemeo digital."""

    def __init__(self, window_name="Digital Twin", width=1280, height=720):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)
        self.first_frame = True
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.current_geometries = []

    def update(self, scene_pcd, objects):
        if scene_pcd is None:
            return

        self.vis.clear_geometries()
        self.current_geometries = []

        scene_copy = copy.deepcopy(scene_pcd)
        self.current_geometries.append(scene_copy)
        self.vis.add_geometry(scene_copy, reset_bounding_box=self.first_frame)

        coord_copy = copy.deepcopy(self.coordinate_frame)
        self.current_geometries.append(coord_copy)
        self.vis.add_geometry(coord_copy)

        for obj in objects:
            mesh = self._create_mesh(obj)
            if mesh is None:
                continue
            self.current_geometries.append(mesh)
            self.vis.add_geometry(mesh)

        self.vis.poll_events()
        self.vis.update_renderer()
        self.first_frame = False

    def _create_mesh(self, obj_info):
        shape = obj_info["shape"]
        dims = obj_info["dimensions"]
        pose = obj_info["pose"]
        color = obj_info["color"]

        if pose is None or dims is None:
            return None

        if shape == "sphere":
            radius = float(np.mean(dims) * 0.5)
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        elif shape == "cylinder":
            mesh = self._build_cylinder_mesh(dims)
        elif shape == "box":
            mesh = self._build_box_mesh(dims)
        else:
            mesh = self._build_box_mesh(dims)

        mesh.paint_uniform_color(color.tolist())
        mesh.compute_vertex_normals()
        mesh.transform(pose)
        return mesh

    def _build_box_mesh(self, dims):
        width, height, depth = dims
        mesh = o3d.geometry.TriangleMesh.create_box(width, height, depth)
        mesh.translate([-width * 0.5, -height * 0.5, -depth * 0.5])
        return mesh

    def _build_cylinder_mesh(self, dims):
        idx_sorted = np.argsort(dims)
        small1, small2, large = dims[idx_sorted]
        height_axis = idx_sorted[-1]
        radius = float((small1 + small2) * 0.25)
        height = float(large)

        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        mesh.translate([0.0, 0.0, -height * 0.5])

        if height_axis == 0:
            rot = o3d.geometry.get_rotation_matrix_from_xyz([0.0, math.pi / 2.0, 0.0])
            mesh.rotate(rot, center=(0.0, 0.0, 0.0))
        elif height_axis == 1:
            rot = o3d.geometry.get_rotation_matrix_from_xyz([-math.pi / 2.0, 0.0, 0.0])
            mesh.rotate(rot, center=(0.0, 0.0, 0.0))
        return mesh

    def close(self):
        self.vis.destroy_window()


if __name__ == "__main__":
    pipeline = PoseEstimationPipeline(voxel_size=0.01)
    visualizer = DigitalTwinVisualizer()

    try:
        while True:
            objects, scene = pipeline.run_single_frame()

            if objects:
                print(f"objetos detectados: {len(objects)}")
                for obj in objects:
                    pose = obj["pose"]
                    center = pose[:3, 3] if pose is not None else np.zeros(3)
                    dims = obj["dimensions"]
                    print(
                        f"  - classe: {obj['class_name']} | forma: {obj['shape']} | "
                        f"dimensoes: {np.round(dims, 3)} | centro: {np.round(center, 3)} | "
                        f"conf: {obj['confidence']:.2f}"
                    )
            else:
                print("nenhum objeto com geometria consistente detectado neste quadro.")

            visualizer.update(scene, objects)

    except KeyboardInterrupt:
        print("interrupcao do usuario. encerrando.")
    finally:
        visualizer.close()
        pipeline.stop()
