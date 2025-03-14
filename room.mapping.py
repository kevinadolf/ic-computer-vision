import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import math

def capture_point_cloud(pipeline, intrinsics, depth_scale):
    """
    captura um frame de profundidade do LIDAR L515 e gera uma nuvem (ou matriz) de pontos
    
    Parametros:
    - pipeline: objeto do RealSense pipeline (já iniciado).
    - intrinsics: parâmetros intrínsecos da câmera.
    - depth_scale: escala de conversão dos valores de profundidade.
    
    output:
    - pcd (point cloud data): objeto open3d.geometry.PointCloud com a nuvem de pontos gerada.
    """
    # aguarda a captura dos frames (sincroniza o frame)
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None

    # converte o frame de profundidade para um array numpy
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # cria uma imagem Open3D a partir do array de profundidade
    o3d_depth = o3d.geometry.Image(depth_image)
    
    # converte a imagem de profundidade em uma nuvem de pontos
    # O parametro depth_trunc define um limite de distância,
    # precisamos ajustar esse valor de acordo com o cômodo
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
        depth_trunc=3.0,
        stride=1
    )
    return pcd

def rotation_matrix_y(angle_deg):
    """
    cria uma matriz de transformacao 4x4 para rotação em torno do eixo Y
    
    parametros:
    - angle_deg (angle degrees): angulo de rotação em graus
    
    retorno:
    - matriz de transformacao 4x4 (numpy.array) correspondente a rotacao
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)
    # matriz de rotacao 4x4 em torno do eixo Y
    R = np.array([
        [cos_val, 0, sin_val, 0],
        [0,       1, 0,       0],
        [-sin_val,0, cos_val, 0],
        [0,       0, 0,       1]
    ])
    return R

def main():
    # configuracao do pipeline do RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # inicializa o pipeline e obtem os parametros do sensor
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    video_stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = video_stream_profile.get_intrinsics()
    
    # lista para armazenar os pontos de todas as capturas
    all_points = []
    
    # definir quantas capturas serão feitas para completar 360°
    num_capturas = 36  # por exemplo, 36 capturas a cada 10°,
                       # temos que ajustar de acordo com o angulo de rotacao da base giratoria
    incremento_angulo = 360 / num_capturas
    
    print("Iniciando o mapeamento 360° do comodo...\n")
    for i in range(num_capturas):
        # calcula o angulo atual de captura
        angulo_atual = i * incremento_angulo
        print(f"Capturando nuvem de pontos no ângulo: {angulo_atual:.2f}°")
        
        # captura a nuvem de pontos atual
        pcd = capture_point_cloud(pipeline, intrinsics, depth_scale)
        if pcd is None:
            print("Falha na captura do frame de profundidade. Tentando novamente...")
            continue
        
        # aplica a rotacao correspondente ao angulo atual
        transform = rotation_matrix_y(angulo_atual)
        pcd.transform(transform)
        
        # converte a nuvem de pontos para um array e adiciona a lista global
        points = np.asarray(pcd.points)
        all_points.append(points)
        
        # espera um tempo ate que base gire para o próximo ângulo, PASSIVEL DE AJUSTE!
        time.sleep(1)
    
    # concatena todas as nuvens de pontos em uma única nuvem global
    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        global_pcd = o3d.geometry.PointCloud()
        global_pcd.points = o3d.utility.Vector3dVector(all_points)
        
        print("Visualizando o mapeamento completo...")
        o3d.visualization.draw_geometries([global_pcd])

        # salva o mapeamento final em um arquivo .ply
        o3d.io.write_point_cloud("mapa_360.ply", global_pcd)
        print("O mapeamento foi salvo como 'mapa_360.ply'")
    else:
        print("Nenhuma nuvem de pontos foi capturada.")
    
    # encerra o pipeline do RealSense
    pipeline.stop()

if __name__ == '__main__':
    main()
