import cv2
import os

# 加载预训练的分类器模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

def apply_mosaic(image, x, y, w, h, neighbor=20):
    """
    对图像的指定区域应用马赛克效果
    """
    mosaic_area = image[y:y+h, x:x+w]
    mosaic_area = cv2.resize(mosaic_area, (w // neighbor, h // neighbor), interpolation=cv2.INTER_LINEAR)
    mosaic_area = cv2.resize(mosaic_area, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic_area
    return image

def detect_and_mosaic(image_path):
    """
    检测人脸和车牌号并应用马赛克效果
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read the image file: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        image = apply_mosaic(image, x, y, w, h)

    # 检测车牌号
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in plates:
        image = apply_mosaic(image, x, y, w, h)

    # 再次检测车牌号以确保所有车牌都被识别
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in plates:
        image = apply_mosaic(image, x, y, w, h)

    

    return image

def process_multiple_images(image_paths):
    """
    处理多张图片，显示并在关闭后删除图片文件
    """
    for index, image_path in enumerate(image_paths):
        result_image = detect_and_mosaic(image_path)
        output_path = f'data/outputs/output_{index}.png'
        cv2.imwrite(output_path, result_image)
        
        # 计算新尺寸以适应屏幕大小
        screen_res = 1280, 720  # 适合大多数现代屏幕
        scale_width = screen_res[0] / result_image.shape[1]
        scale_height = screen_res[1] / result_image.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(result_image.shape[1] * scale)
        window_height = int(result_image.shape[0] * scale)

        # 调整图片大小以适应屏幕
        result_image = cv2.resize(result_image, (window_width, window_height))

        # 显示处理后的图片
        cv2.imshow('Mosaic Result', result_image)
        cv2.waitKey(0)  # 等待用户按键
        cv2.destroyAllWindows()  # 关闭所有窗口
        
        # # 删除输出文件
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Deleted the output file: {output_path}")

def main():
    # 图片路径列表
    image_paths = [
        # 'data/inputs/image.jpg',
        'data/inputs/image2.jpg',
        'data/inputs/image3.jpg',
        # 'data/inputs/image4.jpg',
        # 'data/inputs/image5.jpg',
    ]
    
    # 处理多张图片
    process_multiple_images(image_paths)

if __name__ == "__main__":
    main()





