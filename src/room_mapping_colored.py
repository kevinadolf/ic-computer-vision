import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import math

def capture_point_cloud(pipeline, intrinsics, depth_scale):
    """
    Captura um frame de profundidade e cor do LIDAR L515 e gera uma nuvem de pontos colorida.
    
    Parâmetros:
      - pipeline: objeto do RealSense pipeline (já iniciado).
      - intrinsics: parâmetros intrínsecos da câmera.
      - depth_scale: escala de conversão dos valores de profundidade.
    
    Retorno:
      - pcd: objeto open3d.geometry.PointCloud com a nuvem de pontos colorida gerada.
    """
    # Aguarda a captura dos frames (sincroniza os streams de profundidade e cor)
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None

    # Converte o frame de profundidade para um array numpy
    depth_image = np.asanyarray(depth_frame.get_data())
    # Converte o frame de cor para um array numpy (formato BGR)
    color_image = np.asanyarray(color_frame.get_data())
    
    # Cria imagens Open3D a partir dos arrays
    o3d_depth = o3d.geometry.Image(depth_image)
    o3d_color = o3d.geometry.Image(color_image)
    
    # Cria um objeto de câmera Pinhole com os parâmetros intrínsecos
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, 
        intrinsics.height, 
        intrinsics.fx, 
        intrinsics.fy, 
        intrinsics.ppx, 
        intrinsics.ppy
    )
    
    # Gera a nuvem de pontos colorida a partir dos frames de cor e profundidade.
    # Utiliza create_from_color_and_depth para associar cada ponto com sua cor.
    pcd = o3d.geometry.PointCloud.create_from_color_and_depth(
        o3d_color, 
        o3d_depth, 
        pinhole_camera_intrinsic,
        np.identity(4),
        depth_scale=1.0/depth_scale, 
        depth_trunc=3.0,
        stride=1
    )
    return pcd

def rotation_matrix_y(angle_deg):
    """
    Cria uma matriz de transformação 4x4 para rotação em torno do eixo Y.
    
    Parâmetros:
      - angle_deg: ângulo de rotação em graus.
    
    Retorno:
      - Matriz de transformação 4x4 (numpy.array) correspondente à rotação.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)
    # Matriz de rotação 4x4 em torno do eixo Y (assumindo que Y seja o eixo vertical)
    R = np.array([
        [cos_val, 0, sin_val, 0],
        [0,       1, 0,       0],
        [-sin_val,0, cos_val, 0],
        [0,       0, 0,       1]
    ])
    return R

def main():
    # Configuração do pipeline do RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    # Habilita o stream de profundidade
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # Habilita o stream de cor
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Inicializa o pipeline e obtém os parâmetros do sensor
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    video_stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = video_stream_profile.get_intrinsics()
    
    # Lista para armazenar os pontos de todas as capturas
    all_points = []
    
    # Definir quantas capturas serão feitas para completar 360°
    num_capturas = 36  # por exemplo, 36 capturas a cada 10°
    incremento_angulo = 360 / num_capturas
    
    print("Iniciando o mapeamento 360°...")
    for i in range(num_capturas):
        # Calcula o ângulo atual de captura
        angulo_atual = i * incremento_angulo
        print(f"Capturando nuvem de pontos no ângulo: {angulo_atual:.2f}°")
        
        # Captura a nuvem de pontos atual (com cor)
        pcd = capture_point_cloud(pipeline, intrinsics, depth_scale)
        if pcd is None:
            print("Falha na captura do frame. Tentando novamente...")
            continue
        
        # Aplica a rotação correspondente ao ângulo atual
        transform = rotation_matrix_y(angulo_atual)
        pcd.transform(transform)
        
        # Converte a nuvem de pontos para um array e adiciona à lista global
        points = np.asarray(pcd.points)
        # Também armazena as cores associadas (se necessário para processamento posterior)
        colors = np.asarray(pcd.colors)
        
        # Cria uma nova nuvem de pontos com cor para armazenamento
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points)
        pcd_temp.colors = o3d.utility.Vector3dVector(colors)
        all_points.append(pcd_temp)
        
        # Aguarda um tempo para que a base gire para o próximo ângulo (ajuste conforme necessário)
        time.sleep(1)
    
    # Combina todas as nuvens de pontos em uma única nuvem global
    if all_points:
        global_pcd = all_points[0]
        for pcd in all_points[1:]:
            global_pcd += pcd
        
        print("Visualizando o mapeamento completo...")
        o3d.visualization.draw_geometries([global_pcd])
        
        # Salva o mapeamento final em um arquivo PLY com informações de cor
        o3d.io.write_point_cloud("mapa_360_colorido.ply", global_pcd)
        print("O mapeamento foi salvo como 'mapa_360_colorido.ply'")
    else:
        print("Nenhuma nuvem de pontos foi capturada.")
    
    # Encerra o pipeline do RealSense
    pipeline.stop()

if __name__ == '__main__':
    main()
