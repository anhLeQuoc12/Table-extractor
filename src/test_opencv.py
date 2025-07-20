import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
from PIL import Image

img = cv2.imread("../input_images/page_1.png")
model = YOLO("yolo11n.pt")
result = model("../input_images/page_5.png")[0]
img_bgr = result.plot()
img_rgb = Image.fromarray(img_bgr[..., ::-1])
result.show()
