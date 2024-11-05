import math
import re

import numpy as np
from flask import Flask
import requests
from shapely.geometry import Polygon, Point
from PIL import Image
import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint, request, jsonify, send_from_directory
from flask_cors import CORS
from itertools import combinations
import utils
from extensions import db
from datasets_api import Data, Dataset
import shutil
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


def create_dataset_folders_rec(dataset_name):
    dataset_folder = os.path.join("annotation_Dataset_rec", dataset_name)
    images_folder = os.path.join(dataset_folder, "images")
    label_file_path = os.path.join(dataset_folder, "label.txt")

    # 创建目录
    os.makedirs(images_folder, exist_ok=True)

    return images_folder, label_file_path


# 计算坐标的质心
def calculate_centroid(coordinates_str):
    coords = [int(coord) for coord in coordinates_str.split(',')]
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    # x_coords = [p[0] for p in points]
    # y_coords = [p[1] for p in points]
    # centroid_x = sum(x_coords) / len(points)
    # centroid_y = sum(y_coords) / len(points)
    return points


# 计算多边形交集，并保留 poly2 中的边，凑够 4 个点，选择 poly1 中能最大化面积的点
def calculate_intersection(poly1, poly2):
    # 创建 Polygon 对象
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)

    # 计算多边形交集
    intersection = polygon1.intersection(polygon2)

    # 如果交集为空，返回 None
    if intersection.is_empty:
        return None

    # 获取交集的最小外接矩形，确保用 4 个点框住
    intersection_rectangle = intersection.minimum_rotated_rectangle
    intersection_coords = list(intersection_rectangle.exterior.coords)[:4]  # 保留 4 个点

    # 将坐标转换为整数像素坐标
    pixel_coords = [(int(coord[0]), int(coord[1])) for coord in intersection_coords]

    return pixel_coords


# 切割图片并保存
def save_cropped_image(original_image_path, matched_annotations):
    num = 0
    for i in matched_annotations:
        image = Image.open(original_image_path)

        # 获取多边形的最小边界框并切割图片
        polygon = Polygon(i['points'])
        bounds = polygon.bounds  # 获取多边形的最小边界框 (x_min, y_min, x_max, y_max)
        cropped_image = image.crop(bounds)

        # 如果图片是 RGBA 模式，将其转换为 RGB 模式
        if cropped_image.mode == 'RGBA':
            cropped_image = cropped_image.convert('RGB')
        temp = original_image_path.split('/')[-1].split('.')
        img_filename = str(temp[0]) + '_' + str(num) + str(temp[1])
        # 保存图片为 JPEG 格式
        cropped_image.save(os.path.join("annotation_Dataset_rec", original_image_path.split('/')[1], img_filename),
                           format='JPEG')
        num += 1


def round_points(points):
    return [[math.ceil(point[0]), math.ceil(point[1])] for point in points]


def add_slash_to_unicode(match):
    # 获取匹配的Unicode字符
    unicode_char = match.group(0)
    # 将字符前添加斜杠
    stra = f"/{unicode_char}"
    return stra.encode('utf-8').decode('unicode_escape')


def convert_unicode(file_path):
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     content = f.read()
    # converted_content = re.sub(r'\\u[0-9a-fA-F]{4}', add_slash_to_unicode, content)
    #
    # with open(file_path, 'w', encoding='utf-8') as f:
    #     f.write(converted_content)

    valid_lines = []

    # 读取文件并处理
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取 transcription 和 points 部分
            image_path, json_data = line.strip().split('\t')
            json_obj = json.loads(json_data)
            # 对 points 中的小数进行向上取整
            json_obj[0]['points'] = round_points(json_obj[0]['points'])
            valid_lines.append(f"{image_path}\t{json.dumps(json_obj)}\n")
    os.remove(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    print(f"文件 {file_path} 已更新。")


def convert_total(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉行末的换行符
            line = line.strip()
            # 使用制表符分割行
            filename, json_str = line.split('\t')
            # 将JSON字符串转换为字典对象
            json_obj = json.loads(json_str)
            # 如果文件名不在字典中，初始化一个空数组
            if filename not in data_dict:
                data_dict[filename] = []
            # 将JSON对象添加到对应的数组中
            data_dict[filename].append(json_obj[0])
    # 将字典写入输出文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for filename, json_list in data_dict.items():
            # 创建一个新的行，包含文件名和对应的JSON数组
            line = f"{filename}\t{json.dumps(json_list)}\n"
            f.write(line)


def convert_to_relative_pixel_position(detected_polygon, intersection_points):
    """
    将 intersection_points 转换为相对于 detected_polygon 的相对像素位置
    """
    # 以 detected_polygon 的第一个点为参考点（左上角）
    reference_point = detected_polygon[0]

    # 对每个 intersection_point 进行坐标变换
    relative_points = [[point[0] - reference_point[0], point[1] - reference_point[1]] for point in intersection_points]

    return relative_points


# 主函数：处理标注数据，生成新的数据集
def process_annotations(dataset_name, original_image_path, matched_annotations):
    images_folder, label_file_path = create_dataset_folders(dataset_name)
    shutil.copy(original_image_path, images_folder)

    # 打开标注文件
    with open(label_file_path, 'a') as label_file:
        for annotation in matched_annotations:
            # text_polygon = annotation['points']  # 标注信息的多边形
            # detected_polygon = annotation['polygon']  # 识别的多边形
            #
            # intersection_points = calculate_intersection(text_polygon, detected_polygon)
            # if not intersection_points:
            #     continue

            img_filename = original_image_path.split('/')[-1]

            # 切割并保存图片

            annotation_data = {
                "transcription": annotation['text'],
                # "points": convert_to_relative_pixel_position(detected_polygon, intersection_points)
                "points": annotation['points']
            }

            label_entry = f"{img_filename}\t[{json.dumps(annotation_data)}]\n"
            label_file.write(label_entry)
    # convert_unicode(label_file_path)


def process_annotations_rec(dataset_name, original_image_path, matched_annotations):
    images_folder, label_file_path = create_dataset_folders_rec(dataset_name)
    shutil.copy(original_image_path, os.path.join("annotation_Dataset_rec", dataset_name))

    # 打开标注文件
    with open(label_file_path, 'a') as label_file:
        num = 1
        for annotation in matched_annotations:
            # text_polygon = annotation['points']  # 标注信息的多边形
            # detected_polygon = annotation['polygon']  # 识别的多边形
            #
            # intersection_points = calculate_intersection(text_polygon, detected_polygon)
            # if not intersection_points:
            #     continue
            temp = original_image_path.split('/')[-1].split('.')
            img_filename = str(temp[0]) + '_' + str(num) + str(temp[1])
            # 切割并保存图片
            label_entry = f"{img_filename}\t{annotation['text']}\n"
            label_file.write(label_entry)
            num += 1

    # convert_unicode(label_file_path)


def r_and_p(dataset_name, image_name):
    if not image_name or not dataset_name:
        return jsonify({"error": "Missing parameters"}), 400
    pic_path = os.path.join('datasets', dataset_name, image_name)
    base64_image = utils.image_to_base64(pic_path)

    # 准备POST请求的Payload
    payload = {
        "image": base64_image
    }
    # 发送POST请求到Flask服务
    # response = requests.post('http://127.0.0.1:5000/api/recognize', json=payload)
    # data = response.json()
    #
    # polygons = data['polygons']

    # 获取数据库中的标注信息
    image_name = image_name.split('.')[0]
    dataset = Dataset.query.filter_by(name=dataset_name).first()
    dataset_id = dataset.id
    data_entries = Data.query.filter_by(dataset_id=dataset_id).filter(
        Data.image_path.contains(str(image_name[9:]))).all()
    # result = []
    # for i in data_entries:
    #     if image_name[9:] in i.image_path.split('/')[-1]:
    #         result.append(i)
    annotations = data_entries

    if not annotations:
        return jsonify({"error": "No annotations found for this image"}), 404

    # 匹配标注信息和多边形
    # matched_annotations = []
    # for annotation in annotations:
    #     centroid = calculate_centroid(annotation.coordinates)
    #
    #     for polygon in polygons:
    #         polygon_coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
    #         if Polygon(polygon_coords).contains(Point(centroid)):
    #             coords_list = [float(c) for c in annotation.coordinates.split(',')]
    #             points = [(coords_list[i], coords_list[i + 1]) for i in range(0, len(coords_list), 2)]
    #             matched_annotations.append({
    #                 'text': annotation.text,
    #                 'polygon': polygon_coords,
    #                 'points': points
    #                 # 标注多边形
    #             })
    #             break

    # 直接整理标注信息
    matched_annotations = []
    for annotation in annotations:
        centroid = calculate_centroid(annotation.coordinates)

        matched_annotations.append({
            'text': annotation.text,
            'polygon': "",
            'points': centroid
            # 标注多边形
        })
    # 处理匹配后的数据并生成新的数据集
    process_annotations(dataset_name, pic_path, annotations)


def r_and_p_rec(dataset_name, image_name):
    if not image_name or not dataset_name:
        return jsonify({"error": "Missing parameters"}), 400
    pic_path = os.path.join('datasets', dataset_name, image_name)

    # 获取数据库中的标注信息
    image_name = image_name.split('.')[0]
    dataset = Dataset.query.filter_by(name=dataset_name).first()
    dataset_id = dataset.id
    data_entries = Data.query.filter_by(dataset_id=dataset_id).filter(
        Data.image_path.contains(str(image_name[9:]))).all()
    # result = []
    # for i in data_entries:
    #     if image_name[9:] in i.image_path.split('/')[-1]:
    #         result.append(i)
    annotations = data_entries

    matched_annotations = []
    for annotation in annotations:
        centroid = calculate_centroid(annotation.coordinates)

        matched_annotations.append({
            'text': annotation.text,
            'polygon': "",
            'points': centroid
            # 标注多边形
        })
    # 处理匹配后的数据并生成新的数据集
    process_annotations_rec(dataset_name, pic_path, matched_annotations)
    return  pic_path, matched_annotations



# API：发送图片到识别 API，获取多边形列表，匹配标注信息并生成数据集
@solve_data_api.route('/recognize_and_process', methods=['POST'])
def recognize_and_process():
    dataset_name = request.form.get('dataset_name')
    filelist = os.listdir("./datasets/" + dataset_name)
    image_names = []
    for i in filelist:
        if i.endswith('.jpg') or i.endswith('.png'):
            image_names.append(i)
    for i in image_names:
        print(i)
        r_and_p(dataset_name, i)

    convert_total("./annotation_Dataset/" + dataset_name + "/label.txt")

    return jsonify({"success": True, "message": "Dataset created and annotations processed"}), 2003


@solve_data_api.route('/recognize_and_process_rec', methods=['POST'])
def recognize_and_p():
    dataset_name = request.form.get('dataset_name')
    filelist = os.listdir("./datasets/" + dataset_name)
    image_names = []
    params = []
    for i in filelist:
        if i.endswith('.jpg') or i.endswith('.png'):
            image_names.append(i)
    for i in image_names:
        print(i)
        pic_path ,matched_annotations = r_and_p_rec(dataset_name, i)
        params.append([os.path.join("annotation_Dataset_rec", dataset_name, pic_path.split('/')[-1]),matched_annotations])
    utils.start_processing_with_multiprocessing(params)

    return jsonify({"success": True, "message": "Dataset created and annotations processed"}), 2003
