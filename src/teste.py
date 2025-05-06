import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if not devices:
    raise RuntimeError("Nenhum dispositivo RealSense detectado. Verifique a conexão!")

for dev in devices:
    print(f"Dispositivo detectado: {dev.get_info(rs.camera_info.name)}")