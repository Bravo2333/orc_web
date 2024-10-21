from flask import Flask
import requests
from shapely.geometry import Polygon,Point
from PIL import Image
import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint, request, jsonify, send_from_directory
from flask_cors import CORS

import utils
from extensions import db
from datasets_api import Data, Dataset
import os
# 创建蓝图
solve_data_api = Blueprint('solve_data_api', __name__)
CORS(solve_data_api)


# 创建数据集目录
def create_dataset_folders(dataset_name):
    dataset_folder = os.path.join("annotation_Dataset", dataset_name)
    images_folder = os.path.join(dataset_folder, "images")
    label_file_path = os.path.join(dataset_folder, "label.txt")

    # 创建目录
    os.makedirs(images_folder, exist_ok=True)

    return images_folder, label_file_path


# 计算坐标的质心
def calculate_centroid(coordinates_str):
    coords = [float(coord) for coord in coordinates_str.split(',')]
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return centroid_x, centroid_y


# 计算多边形交集
def calculate_intersection(poly1, poly2):
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    intersection = polygon1.intersection(polygon2)
    if intersection.is_empty:
        return None
    return list(intersection.exterior.coords)


# 切割图片并保存
def save_cropped_image(original_image_path, polygon_coords, image_save_path):
    image = Image.open(original_image_path)
    polygon = Polygon(polygon_coords)
    bounds = polygon.bounds  # 获取多边形的最小边界框
    cropped_image = image.crop(bounds)
    cropped_image.save(image_save_path)


# 主函数：处理标注数据，生成新的数据集
def process_annotations(dataset_name, original_image_path, matched_annotations):
    images_folder, label_file_path = create_dataset_folders(dataset_name)

    # 打开标注文件
    with open(label_file_path, 'a') as label_file:
        img_count = 1

        for annotation in matched_annotations:
            text_polygon = annotation['points']  # 标注信息的多边形
            detected_polygon = annotation['polygon']  # 识别的多边形

            intersection_points = calculate_intersection(text_polygon, detected_polygon)
            if not intersection_points:
                continue

            img_filename = f"img_{img_count}.jpg"
            image_save_path = os.path.join(images_folder, img_filename)

            # 切割并保存图片
            save_cropped_image(original_image_path, detected_polygon, image_save_path)

            annotation_data = {
                "transcription": annotation['text'],
                "points": intersection_points
            }

            label_entry = f"images/{img_filename}\t[{json.dumps(annotation_data)}]\n"
            label_file.write(label_entry)

            img_count += 1


# API：发送图片到识别 API，获取多边形列表，匹配标注信息并生成数据集
@solve_data_api.route('/recognize_and_process', methods=['POST'])
def recognize_and_process():
    image_name = request.form.get('image_name')
    dataset_name = request.form.get('dataset_name')

    if not image_name or not dataset_name:
        return jsonify({"error": "Missing parameters"}), 400
    pic_path = os.path.join('datasets',dataset_name,image_name)
    base64_image = utils.image_to_base64(pic_path)

    # 准备POST请求的Payload
    payload = {
        "image": base64_image
    }
    # 发送POST请求到Flask服务
    response = requests.post('http://127.0.0.1:5000/api/recognize', json=payload)
    data = response.json()

    polygons = data['polygons']

    # 获取数据库中的标注信息
    image_name = image_name.split('.')[0]
    dataset = Dataset.query.filter_by(name=dataset_name).first()
    dataset_id = dataset.id
    data_entries = Data.query.filter_by(dataset_id=dataset_id).all()
    result = []
    for i in data_entries:
        if image_name[9:] in i.image_path.split('/')[-1]:
            result.append(i)
    annotations = result

    if not annotations:
        return jsonify({"error": "No annotations found for this image"}), 404

    # 匹配标注信息和多边形
    matched_annotations = []
    for annotation in annotations:
        centroid = calculate_centroid(annotation.coordinates)

        for polygon in polygons:
            polygon_coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            if Polygon(polygon_coords).contains(Point(centroid)):
                matched_annotations.append({
                    'text': annotation.text,
                    'polygon': polygon_coords,
                    'points': [(float(p[0]), float(p[1])) for p in Polygon(annotation.coordinates).exterior.coords]
                    # 标注多边形
                })
                break

    # 处理匹配后的数据并生成新的数据集
    process_annotations(dataset_name, pic_path, matched_annotations)

    return jsonify({"success": True, "message": "Dataset created and annotations processed"}), 200

