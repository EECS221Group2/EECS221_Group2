import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

yolov5_path = './yolov5'
sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes as scale_coords
from utils.augmentations import letterbox

classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", 
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
    "hair drier", "toothbrush"
]


def load_model(weights_path='yolov5s.pt', device='cpu'):
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    return model

folder_path = './'  # 设置你的图片文件夹路径

def perform_cleanup_tasks():
    print("Performing cleanup tasks...")
    # 这里可以添加关闭数据库连接、释放资源或其他清理代码
    print("Cleanup completed.")

def process_folder(folder_path, model, classes):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 检查文件是否为图像文件
            img_path = os.path.join(folder_path, filename)
            original_image = cv2.imread(img_path)
            if original_image is None:
                print(f"Failed to load image {img_path}")
                continue

            processed_image = process_image(original_image)  # 应用图像预处理
            output_image = detect_and_mosaic(model, processed_image, original_image, classes)  # 应用马赛克并检测

            # 显示处理后的图像
            cv2.imshow("Mosaic Applied", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 保存处理后的图像到指定路径
            output_path = os.path.join(folder_path, f"mosaic_{filename}")
            cv2.imwrite(output_path, output_image)

    print("All images have been processed.")


    # 执行其他任务，如清理工作或关闭资源
    perform_cleanup_tasks()

def process_image(img):
    img = letterbox(img, new_shape=640)[0]  # 调整图像大小并添加填充
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=3):
    """Draw one bounding box on image img."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color[::-1]  # BGR to RGB
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
def detect_and_mosaic(model, img, original_image, classes):
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.3, 0.4, classes=[0], agnostic=False)  # 确定检测人类的类别索引为0

    for i, det in enumerate(pred):  # 遍历每个检测结果
        if len(det):
            # 从img_size调整到原始尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()

            for *xyxy, conf, cls in det:
                if int(cls) == 0:  # 检测到的类别为人
                    apply_mosaic_to_person(original_image, xyxy)  # 对人应用马赛克

    return original_image

def apply_mosaic_to_person(img, bbox, neighborhood=15):
    x1, y1, x2, y2 = [int(x) for x in bbox]  # 转换为整数
    roi = img[y1:y2, x1:x2]  # 提取感兴趣区域
    h, w = roi.shape[:2]
    if h > 0 and w > 0:  # 确保ROI非空
        roi_small = cv2.resize(roi, (max(1, w // neighborhood), max(1, h // neighborhood)), interpolation=cv2.INTER_LINEAR)
        roi_large = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = roi_large  # 将马赛克区域替换回原图

model = load_model()
process_folder(folder_path, model, classes)

# 加载图像并处理
original_image = cv2.imread('./')
processed_image = process_image(original_image)
output_image = detect_and_mosaic(model, processed_image, original_image, classes)

# 显示或保存结果
cv2.imshow("Mosaic Applied", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output_with_mosaic.jpg", output_image)
