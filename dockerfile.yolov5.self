# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
RUN apt update && apt install --no-install-recommends -y git python3.8 python3-pip
RUN git clone https://github.com/cyhkelvin/yolov5.git /home/yolov5
RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Taipei apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

WORKDIR /home/yolov5
RUN pip install --no-cache -r requirements.prune.txt
RUN mkdir runs
CMD ["python3", "detect_server_simple.py"]
## run with trained model: --weights runs/train/merged/weights/best.pt
# CMD ["python", "detect_server_simple.py", "--weights", "runs/train/merged/weights/best.pt"]

# docker build -t aisdit:object_detection_server -f dockerfile.yolov5.self .
# sudo docker run -v <local path>:/home/yolov5/runs --name <name> -p <port>:8888 aisdit:object_detection_server
# python3.7 client_simple.py --source <file in local path>