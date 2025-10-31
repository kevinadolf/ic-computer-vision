import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as ScipyRotation
import copy

# --- modulo 1: interface da camera ---
class RealSenseCamera:
    """
    essa classe cuida de tudo relacionado a camera intel realsense.
    """
    def __init__(self, width=1280, height=720, fps=30):
        print("inicializando a camera realsense...")
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(self.config)
        
        # pega o fator de escala para a profundidade
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # o alinhamento e super importante para garantir que a imagem de cor e a de profundidade coincidam
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.intrinsics = self._get_o3d_intrinsics()
        print("camera realsense inicializada com sucesso.")

    def _get_o3d_intrinsics(self):
        """pega os parametros intrinsecos da camera e converte para o formato do open3d."""
        video_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = video_profile.get_intrinsics()
        return o3d.camera.PinholeCameraIntrinsic(
            width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy
        )

    def get_frames(self):
        """espera por um novo conjunto de imagens (cor e profundidade) e retorna elas."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("aviso: quadro de profundidade ou cor nao encontrado.")
                return None, None
                
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return depth_image, color_image
        except RuntimeError as e:
            print(f"erro ao obter quadros: {e}")
            return None, None

    def stop(self):
        """para a camera."""
        print("parando o pipeline da camera.")
        self.pipeline.stop()

# --- modulo 2: detecçao de objetos ---
class ObjectDetector:
    """
    essa classe usa o yolo para encontrar objetos na imagem.
    """
    def __init__(self, model_path='yolov8n.pt'):
        print(f"carregando o modelo yolo do caminho: {model_path}")
        self.model = YOLO(model_path)
        print("modelo yolo carregado com sucesso.")

    def detect(self, image, confidence_threshold=0.5):
        """detecta objetos em uma imagem e retorna uma lista com o que encontrou."""
        results = self.model(image, verbose=False)
        detections =
        for result in results:
            for box in result.boxes:
                if box.conf > confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    detections.append({
                        'class_name': class_name,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf)
                    })
        return detections

# --- modulo 3: funçoes do pipeline de pose ---
class PoseEstimationPipeline:
    """
    aqui a magica acontece. essa classe junta tudo para estimar a pose 6d.
    """
    def __init__(self, model_paths, voxel_size=0.005):
        self.camera = RealSenseCamera()
        self.detector = ObjectDetector()
        self.intrinsics = self.camera.intrinsics
        self.voxel_size = voxel_size
        
        print("carregando e pre-processando modelos 3d...")
        self.models = self._load_models(model_paths)
        print("modelos 3d carregados.")

    def _load_models(self, model_paths):
        """carrega os modelos 3d (cad) dos objetos que queremos encontrar."""
        models = {}
        for class_name, path in model_paths.items():
            print(f"  - carregando '{class_name}' de '{path}'")
            try:
                model_mesh = o3d.io.read_triangle_mesh(path)
                model_mesh.compute_vertex_normals()
            except Exception as e:
                print(f"erro ao carregar o modelo {path}: {e}")
                continue

            # e bom amostrar a malha para ter uma nuvem de pontos uniforme
            model_pcd = model_mesh.sample_points_poisson_disk(number_of_points=5000)
            model_pcd_down = model_pcd.voxel_down_sample(self.voxel_size)
            model_pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
            
            model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                model_pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100))
            
            models[class_name] = {'pcd': model_pcd_down, 'fpfh': model_fpfh, 'full_pcd': model_pcd}
        return models

    def run_single_frame(self):
        """executa uma unica vez o pipeline completo."""
        depth_image, color_image = self.camera.get_frames()
        if depth_image is None or color_image is None:
            return None, None

        # 1. cria a nuvem de pontos da cena inteira
        scene_pcd = self._create_point_cloud_from_frames(depth_image, color_image)
        scene_pcd_down = scene_pcd.voxel_down_sample(self.voxel_size)
        scene_pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

        # 2. detecta os objetos na imagem 2d
        detections = self.detector.detect(color_image)
        
        estimated_poses =
        
        for det in detections:
            class_name = det['class_name']
            if class_name not in self.models:
                continue

            print(f"\nprocessando objeto detectado: {class_name}")

            # 3. recorta o objeto da nuvem de pontos da cena
            object_pcd = self._segment_object_from_bbox(scene_pcd, det['bbox'])
            if object_pcd is None or len(object_pcd.points) < 100:
                print(f"  - segmentacao falhou ou objeto muito pequeno. pulando.")
                continue

            object_pcd_down = object_pcd.voxel_down_sample(self.voxel_size)
            object_pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

            # 4. registro global (uma primeira aproximacao da pose)
            print("  - executando registro global (ransac)...")
            model = self.models[class_name]
            object_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                object_pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100))
            
            coarse_transform = self._execute_global_registration(
                object_pcd_down, model['pcd'], object_fpfh, model['fpfh']
            )

            # 5. registro local (refinamento da pose com icp)
            print("  - executando registro local (icp)...")
            result = self._refine_registration(
                model['full_pcd'], object_pcd, coarse_transform
            )
            
            print(f"  - fitness do icp: {result.fitness:.3f}, rmse: {result.inlier_rmse:.4f}")

            # 6. valida o resultado e guarda se for bom
            if result.fitness > 0.6 and result.inlier_rmse < self.voxel_size:
                print("  - pose estimada com sucesso!")
                estimated_poses.append({
                    'class_name': class_name,
                    'pose': result.transformation,
                    'fitness': result.fitness,
                    'rmse': result.inlier_rmse
                })
            else:
                print("  - a pose estimada nao parece boa o suficiente.")

        return estimated_poses, scene_pcd

    def _create_point_cloud_from_frames(self, depth_image, color_image):
        """cria uma nuvem de pontos open3d a partir das imagens de profundidade e cor."""
        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=self.camera.depth_scale, depth_trunc=3.0, convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsics)
        # a nuvem de pontos vem virada, entao a gente desvira ela
        pcd.transform([,
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       ])
        return pcd

    def _segment_object_from_bbox(self, scene_pcd, bbox):
        """recorta a nuvem de pontos do objeto usando a caixa de deteccao 2d."""
        x1, y1, x2, y2 = bbox
        
        # cria um poligono 2d a partir da caixa de deteccao
        bounding_polygon = np.array([x1, y1, 0],
            [x2, y1, 0],
            [x2, y2, 0],
            [x1, y2, 0]).astype("float64")

        # transforma o poligono 2d num volume 3d para recortar a nuvem de pontos
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Z"
        vol.axis_max = 5.0 # profundidade maxima do recorte
        vol.axis_min = 0.1 # profundidade minima do recorte
        vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
        
        object_pcd = vol.crop_point_cloud(scene_pcd)
        return object_pcd

    def _execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        """executa o registro global com ransac."""
        distance_threshold = self.voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,, o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result.transformation

    def _refine_registration(self, source, target, initial_transform):
        """refina a pose com o icp ponto-a-plano."""
        threshold = self.voxel_size * 0.4
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        return reg_p2p

    def stop(self):
        """para a camera."""
        self.camera.stop()

# --- modulo 4: visualizacao e utilitarios ---
def visualize_poses(scene_pcd, poses, models):
    """mostra a nuvem de pontos da cena com os modelos 3d alinhados."""
    if scene_pcd is None:
        return
        
    geometries = [scene_pcd]
    
    # cria um sistema de coordenadas para ter uma referencia
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=)
    geometries.append(coord_frame)
    
    for pose_info in poses:
        class_name = pose_info['class_name']
        transform = pose_info['pose']
        
        model_pcd = models[class_name]['full_pcd']
        model_copy = copy.deepcopy(model_pcd)
        model_copy.transform(transform)
        
        # pinta o modelo de uma cor diferente para destacar
        model_copy.paint_uniform_color([1.0, 0.706, 0]) # laranja
        geometries.append(model_copy)
        
        # adiciona um sistema de coordenadas para a pose do objeto
        obj_coord_frame = copy.deepcopy(coord_frame).transform(transform)
        geometries.append(obj_coord_frame)
        
    o3d.visualization.draw_geometries(geometries)

# --- ponto de entrada principal ---
if __name__ == "__main__":
    
    # #####################################################################
    # # IMPORTANTE: AQUI E ONDE VOCE DEVE COLOCAR OS SEUS ARQUIVOS CAD     #
    # #####################################################################
    #
    # o dicionario 'model_paths' mapeia o nome da classe que o yolo detecta
    # para o caminho do arquivo do modelo 3d (.ply,.stl,.obj, etc.).
    #
    # exemplo: se o seu yolo foi treinado para detectar 'garrafa', a chave deve ser 'garrafa'
    # e o valor deve ser o caminho para o arquivo 'garrafa.ply'.
    
    model_paths = {
        'cup': './models/cup_model.ply',
        'bottle': './models/bottle_model.ply'
        # adicione mais objetos aqui. por exemplo:
        # 'meu_objeto': './caminho/para/meu_objeto.stl'
    }
    
    # o codigo abaixo verifica se os modelos existem.
    # se nao existirem, ele cria uns cilindros ficticios para o codigo nao quebrar.
    # voce deve substituir esses cilindros pelos seus modelos reais.
    import os
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if 'cup' in model_paths and not os.path.exists(model_paths['cup']):
        print("aviso: modelo de 'cup' nao encontrado. criando um cilindro ficticio.")
        cup_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.08)
        cup_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(model_paths['cup'], cup_mesh)
    if 'bottle' in model_paths and not os.path.exists(model_paths['bottle']):
        print("aviso: modelo de 'bottle' nao encontrado. criando um cilindro ficticio.")
        bottle_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.04, height=0.2)
        bottle_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(model_paths['bottle'], bottle_mesh)


    pipeline = PoseEstimationPipeline(model_paths, voxel_size=0.005)
    
    try:
        while True:
            print("\n--- pressione 'enter' para capturar e processar um novo quadro (ou 'q' para sair) ---")
            if input() == 'q':
                break
                
            poses, scene = pipeline.run_single_frame()
            
            if poses:
                print(f"\nposes estimadas encontradas: {len(poses)}")
                for p in poses:
                    print(f"  - classe: {p['class_name']}, fitness: {p['fitness']:.2f}, rmse: {p['rmse']:.4f}")
                
                visualize_poses(scene, poses, pipeline.models)
            else:
                print("\nnenhuma pose confiavel foi estimada neste quadro.")
                if scene is not None:
                    print("mostrando apenas a nuvem de pontos da cena.")
                    o3d.visualization.draw_geometries([scene])

    except KeyboardInterrupt:
        print("interrupcao do usuario. encerrando.")
    finally:
        pipeline.stop()