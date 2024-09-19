import cv2
import os
import numpy as np
from paddleocr import PaddleOCR
from detfromimage import det
# 初始化 OCR 模型 (支持det检测和rec识别)
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 可以根据需要选择语言，例如'chs', 'en'


# 定义函数：使用多边形信息切割图片
def split_image_by_polygons(image, polygons):
    cropped_images = []
    for idx, polygon in enumerate(polygons):
        # 提取多边形的坐标
        polygon = polygon.reshape((-1, 2))

        # 计算多边形的最小边界矩形，并裁剪图像
        rect = cv2.boundingRect(polygon)
        x, y, w, h = rect
        cropped = image[y:y + h, x:x + w].copy()

        # 保存裁剪后的图片到列表
        cropped_images.append({
            'image': cropped,
            'polygon': polygon,
            'index': idx
        })
    return cropped_images


# 定义函数：使用det模型检测文本位置
def detect_text_positions(cropped_images):
    detected_results = []
    for cropped in cropped_images:
        image = cropped['image']
        success, encoded_image = cv2.imencode('.png', image)

        # 使用 det 模型检测文本位置
        result = ocr.ocr(image, det=True, rec=False)
        detected_results.append({
            'index': cropped['index'],
            'polygon': cropped['polygon'],
            'detected_boxes': result[0],  # 文本位置框
            'image': image  # 保留原始小图
        })
    return detected_results


# 定义函数：根据det模型检测结果裁剪出文本区域，并传递给rec模型进行文本识别
def recognize_text(detected_results):
    recognition_results = []

    for detected in detected_results:
        image = detected['image']
        recognized_texts = []

        for box in detected['detected_boxes']:
            # box格式：[[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (score)]
            # 提取每个文本区域的坐标，并裁剪出对应区域
            poly = np.array(box[0], dtype=np.int32)

            # 创建用于裁剪的掩膜
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(mask, [poly], (255, 255, 255))

            # 在原始小图片上裁剪出文本区域
            cropped_text_area = cv2.bitwise_and(image, mask)
            x_min = np.min(poly[:, 0])
            y_min = np.min(poly[:, 1])
            x_max = np.max(poly[:, 0])
            y_max = np.max(poly[:, 1])
            cropped_text_area = cropped_text_area[y_min:y_max, x_min:x_max]

            # 使用 rec 模型识别文本内容
            rec_result = ocr.ocr(cropped_text_area, det=False, rec=True)

            # 保存识别出的文本
            if rec_result and len(rec_result) > 0:
                recognized_texts.append(rec_result[0][0])  # 取出识别的文本内容

        recognition_results.append({
            'index': detected['index'],
            'polygon': detected['polygon'],
            'texts': recognized_texts
        })

    return recognition_results


# 保存识别结果到文本文档中
def save_recognition_results(recognition_results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in recognition_results:
            f.write(f"Polygon {result['index']}: {result['texts']}\n")


# 示例主流程
def main(image_path, polygons, output_text_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 1. 使用多边形信息分割图片
    cropped_images = split_image_by_polygons(image, polygons)

    # 2. 使用det模型检测小图片中的文本位置
    detected_results = detect_text_positions(cropped_images)

    # 3. 使用rec模型识别文本内容
    recognition_results = recognize_text(detected_results)

    # 4. 将识别结果与多边形位置对应，并存储到文本文件
    save_recognition_results(recognition_results, output_text_path)

    print(f"识别结果已保存到 {output_text_path}")


# 测试用例
if __name__ == '__main__':
    # 示例图片和多边形信息（实际的多边形信息应从外部提供）
    image_path = 'test_image.jpg'
    output_text_path = 'recognized_texts.txt'

    # 假设有两个多边形，4个点定义一个多边形
    polygons = [
        # 多边形1: 四个顶点坐标
        [[50, 50], [150, 50], [150, 150], [50, 150]],
        # 多边形2: 四个顶点坐标
        [[200, 200], [300, 200], [300, 300], [200, 300]]
    ]
    polygons = [np.array(polygon) for polygon in polygons]  # 将其转换为numpy格式

    # 执行主流程
    main(image_path, polygons, output_text_path)
