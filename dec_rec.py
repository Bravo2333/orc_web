import base64

import cv2
import numpy as np
import requests

from Det_Rec.detfromimage import det
from Det_Rec.recfromimages import rec


# 初始化 OCR 模型 (支持det检测和rec识别)


# 定义函数：使用多边形信息切割图片
def split_image_by_polygons(image, polygons):
    cropped_images = []
    for idx, polygon in enumerate(polygons):
        # 提取多边形的坐标
        polygon = np.array(polygon, dtype=np.float32)
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
    detinstence = det()
    detinstence.det_init()
    detected_results = []
    num = 0
    for cropped in cropped_images:
        image = cropped['image']
        # cv2.imwrite('./temp/'+str(num)+'temp.png', image)

        # png编码格式
        success, encoded_image = cv2.imencode('.png', image)
        # image_base64 = base64.b64encode(encoded_image).decode('utf-8')
        # print(num,image_base64)
        # 使用 det 模型检测文本位置
        result = detinstence.det_infer(encoded_image.tobytes())
        if result is None:
            continue
        # print(num, result)

        resultpolygon = np.array(result, dtype=np.float32)
        print(resultpolygon.shape)
        # resultpolygon = resultpolygon.reshape((-1, 2))
        # 计算多边形的最小边界矩形，并裁剪图像
        # rect = cv2.boundingRect(resultpolygon)
        # x, y, w, h = rect
        # resultcropped = image[y:y + h, x:x + w].copy()
        # cv2.imwrite('./temp/'+str(num)+'temp.png',resultcropped)

        detected_results.append({
            'index': cropped['index'],
            'polygon': cropped['polygon'],
            'detected_boxes': result,  # 文本位置框
            'image': image  # 保留原始小图 opencv
        })
        num += 1
    return detected_results


# 定义函数：根据det模型检测结果裁剪出文本区域，并传递给rec模型进行文本识别
def recognize_text(detected_results):
    recinstence = rec()
    recinstence.rec_init()
    recognition_results = []

    for detected in detected_results:
        image = detected['image']
        recognized_texts = []
        if detected['detected_boxes'] == None:
            # recognition_results.append({
            #     'index': detected['index'],
            #     'polygon': detected['polygon'].tolist(),
            #     'texts': '识别失败，当前单元格无有效检测区域'
            # })
            continue
        reshaped_polygon = detected['polygon'].reshape((-1, 2))
        area = cv2.contourArea(reshaped_polygon)
        detected_area = cv2.contourArea(np.array(detected['detected_boxes']))
        if detected_area < area / 20:
            success, encoded_image = cv2.imencode('.png', image)
            prefix = "data:image/png;base64,"
            image_base64 = base64.b64encode(encoded_image).decode('utf-8')
            recognition_results.append({
                'index': detected['index'],
                'polygon': detected['polygon'].tolist(),
                'texts': '识别失败，当前单元格检测区域过小',
                'image': prefix + image_base64,
            })
            continue
        # box格式：[[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (score)]
        # 提取每个文本区域的坐标，并裁剪出对应区域
        poly = np.array(detected['detected_boxes'], dtype=np.int32)

        # 创建用于裁剪的掩膜
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [poly], (255, 255, 255))
        print(detected['detected_boxes'])
        # 在原始小图片上裁剪出文本区域
        cropped_text_area = cv2.bitwise_and(image, mask)
        x_min = np.min(poly[:, 0])
        y_min = np.min(poly[:, 1])
        x_max = np.max(poly[:, 0])
        y_max = np.max(poly[:, 1])
        cropped_text_area = cropped_text_area[y_min:y_max, x_min:x_max]
        # 测试切割的最后图片
        # filepath = './temp/'+str(detected['index'])+'.png'
        # cv2.imwrite(filepath,cropped_text_area)
        success, encoded_image = cv2.imencode('.png', cropped_text_area)

        # 使用 rec 模型识别文本内容
        rec_result = recinstence.rec_infer(encoded_image.tobytes())
        if rec_result == "无有效内容":
            continue

        # 保存识别出的文本
        if rec_result and len(rec_result) > 0:
            recognized_texts.append(rec_result)  # 取出识别的文本内容
        prefix = "data:image/png;base64,"
        image_base64 = base64.b64encode(encoded_image).decode('utf-8')
        recognition_results.append({
            'index': detected['index'],
            'polygon': detected['polygon'].tolist(),
            'texts': recognized_texts,
            'image': prefix + image_base64,
        })

    return recognition_results


# 保存识别结果到文本文档中
def save_recognition_results(recognition_results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in recognition_results:
            f.write(f"Polygon {result['index']}: {result['texts']}\n")


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 示例主流程
def getrec_result(image_path, polygons, output_text_path):
    # 读取图片
    tempimage = cv2.imread(image_path)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        # 在图片上绘制多边形，颜色为白色，线条宽度为3
        cv2.polylines(tempimage, [points], isClosed=True, color=(255, 255, 255), thickness=3)
    cv2.imwrite('temp.png', tempimage)
    # 保存无表格线的图片
    image = cv2.imread('temp.png')
    # 1. 使用多边形信息分割图片
    cropped_images = split_image_by_polygons(image, polygons)
    # num = 0
    # for i in cropped_images:
    #     success, encoded_image = cv2.imencode('.png', i['image'])
    #     try:
    #         cv2.imwrite('./temp/'+str(num)+'temp.png',encoded_image)
    #         print('保存成功')
    #     except:
    #         print(str(num)+' 保存失败')
    #     num+=1
    # 2. 使用det模型检测小图片中的文本位置
    detected_results = detect_text_positions(cropped_images)

    # 3. 使用rec模型识别文本内容
    recognition_results = recognize_text(detected_results)
    return recognition_results
    # 4. 将识别结果与多边形位置对应，并存储到文本文件
    # save_recognition_results(recognition_results, output_text_path)
    #
    # print(f"识别结果已保存到 {output_text_path}")


# 测试用例
if __name__ == '__main__':
    # 示例图片和多边形信息（实际的多边形信息应从外部提供）
    image_path = 'C:\\Users\\Administrator\\Desktop\\49d521ba-c711-45c8-93c7-95b15b5e8d7d.png'
    output_text_path = 'recognized_texts.txt'
    base64_image = image_to_base64(image_path)

    # 准备POST请求的Payload
    payload = {
        "image": base64_image
    }
    # 发送POST请求到Flask服务
    response = requests.post('http://127.0.0.1:5000/api/recognize', json=payload)
    data = response.json()
    base64_result_image = data['image_base64']
    # 假设有两个多边形，4个点定义一个多边形
    polygons_list = data['polygons']
    print(polygons_list)
    polygons = [[(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)] for polygon in polygons_list]
    polygons = [np.array(polygon) for polygon in polygons]  # 将其转换为numpy格式

    # 执行主流程
    recognition_results = getrec_result(image_path, polygons, output_text_path)
