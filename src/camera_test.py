import pyrealsense2 as rs

# Inicializa o pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configura o dispositivo para capturar dados do LIDAR
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

# Inicia o pipeline
pipeline.start(config)

print("LIDAR L515 conectado e capturando dados!")

# Para o pipeline após alguns segundos
import time
time.sleep(5)
pipeline.stop()
