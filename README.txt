Por motivos de restricción de tamaño de archivos, no se pudo subir el entorno virtual de YOLO, sin embargo, es un módulo abierto que no requiere ajustes del usuario, siga las siguientes instrucciones para entrenar/probar:

En terminal:

1. python -m venv yolov11_env                      # Para crear la carpeta del entorno virtual
2. pip install ultralytics                         # Para instalar la arquitectura de YOLO
3. Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process   #En caso de ser necesario habilitar permiso de políticas
4. yolov11_env\Scripts\activate                    # Para activar entorno virtual
5. yolo detect train data="ruta del yaml" model="ruta del modelo.pt" epochs=300 imgsz=640 batch=16 device=0 # Comando para empezar entrenamiento
6. yolo predict model="ruta del modelo.pt" source="ruta de la carpeta de imágenes de prueba" show=True # Comando para hacer detecciones en imágenes específicas



Ruta del mejor peso del primer entrenamiento: \runs\detect\train3\weights\best.pt
Ruta del mejor peso del segundo entrenamiento: \runs\detect\train4\weights\best.pt
