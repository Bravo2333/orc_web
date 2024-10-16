import base64

from flask import Blueprint, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from PIL import Image
from io import BytesIO
from data_annotation.textin import CommonOcr
import os
from extensions import db
# 配置数据集存储路径
DATASET_FOLDER = 'datasets'

# 确保 datasets 文件夹存在
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)
# 创建蓝图
datasets_api = Blueprint('datasets_api', __name__)
CORS(datasets_api)
# 初始化 SQLAlchemy，注意这里 db 可能需要从 app.py 引入
# db = SQLAlchemy()
# 数据集表
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    path = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 关联数据集内的图片与标注信息
    data = db.relationship('Data', backref='dataset', lazy=True)

# 数据表，存储图片和标注信息
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)  # 图片路径或文件名
    text = db.Column(db.String(255), nullable=False)  # 文字内容
    is_handwritten = db.Column(db.Boolean, nullable=False)  # 是否为手写
    coordinates = db.Column(db.String(255), nullable=False)  # 坐标数组，存储为字符串格式
    confidence = db.Column(db.Float, nullable=False)  # 置信度
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@datasets_api.route('/pending_images/<dataset_name>', methods=['GET'])
def get_pending_images(dataset_name):
    dataset = Dataset.query.filter_by(name=dataset_name).first()
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404

    # 获取数据集对应的文件夹路径
    dataset_folder = dataset.path

    # 检查文件夹是否存在
    if not os.path.exists(dataset_folder):
        return jsonify({"error": "Dataset folder not found"}), 404

    # 列出该文件夹中的所有图片文件（假设图片格式为 .png 或 .jpg）
    images = [f for f in os.listdir(dataset_folder) if f.endswith(('.png', '.jpg'))]

    # 构建完整的图片路径 (可选)
    # image_paths = [os.path.join(dataset_folder, image) for image in images]

    # 返回图片文件名列表或完整路径列表给前端
    return jsonify(images), 200
# 创建数据集
@datasets_api.route('/create', methods=['POST'])
def create_dataset():
    data = request.get_json()
    dataset_name = data.get('name')

    if not dataset_name:
        return jsonify({"error": "数据集名称不能为空"}), 400

    # 创建数据集存储路径
    dataset_path = os.path.join('datasets', dataset_name)
    if os.path.exists(dataset_path):
        return jsonify({"error": "数据集已存在"}), 400

    os.makedirs(os.path.join(dataset_path, 'images'))

    # 在数据库中创建记录
    new_dataset = Dataset(name=dataset_name, path=dataset_path)
    db.session.add(new_dataset)
    db.session.commit()

    return jsonify({"message": f"数据集 '{dataset_name}' 创建成功"}), 201


# 获取所有数据集
@datasets_api.route('/datasets', methods=['GET'])
def get_datasets():
    datasets = Dataset.query.all()
    response = [{"id": d.id, "name": d.name, "path": d.path, "created_at": d.created_at} for d in datasets]
    return jsonify(response), 200


# 删除数据集
@datasets_api.route('/delete', methods=['POST'])
def delete_dataset():
    data = request.get_json()
    dataset_name = data.get('name')

    dataset = Dataset.query.filter_by(name=dataset_name).first()
    if not dataset:
        return jsonify({"error": "数据集不存在"}), 400

    # 删除文件夹
    dataset_path = dataset.path
    if os.path.exists(dataset_path):
        os.system(f'rm -rf {dataset_path}')

    # 从数据库中删除记录
    Data.query.filter_by(dataset_id=dataset.id).delete()  # 删除所有关联的数据
    db.session.delete(dataset)
    db.session.commit()

    return jsonify({"message": f"数据集 '{dataset_name}' 已删除"}), 200


# 上传图片并标注
@datasets_api.route('/annotate', methods=['POST'])
def annotate_image():
    data = request.json
    dataset_name = data.get('dataset')
    image_data = data.get('image')  # base64 编码的图片

    # 查找对应的数据集
    dataset = Dataset.query.filter_by(name=dataset_name).first()
    if not dataset:
        return jsonify({"error": "数据集不存在"}), 400
    image_data = image_data.split(",")[1]
    # 解码并保存原始图片到 annotation.txt 同级目录下
    image_bytes = base64.b64decode(image_data)
    original_image = Image.open(BytesIO(image_bytes))
    image_format = original_image.format.lower()  # 获取图片格式 (jpeg, png 等)

    # 生成唯一文件名
    image_id = len(os.listdir(os.path.join(dataset.path))) + 1
    original_image_filename = f'original_image_{image_id}.{image_format}'
    original_image_path = os.path.join(dataset.path, original_image_filename)

    # 保存原始图片
    original_image.save(original_image_path)

    # 调用 CommonOcr 进行 OCR 识别，获取标注信息
    ocr_engine = CommonOcr(original_image_path)  # 假设 CommonOcr 接受图片路径
    annotations = ocr_engine.recognize()  # 标注结果，例如：[["文字", 0, [x1, y1, x2, y2...], 置信度]]
    # 使用 Pillow 处理 OCR 识别的标注信息并裁剪文字区域
    for i, annotation in enumerate(annotations):
        text = annotation[0]
        is_handwritten = bool(annotation[1])
        coordinates = annotation[2]  # 例如 [x1, y1, x2, y2, x3, y3, x4, y4]
        confidence = annotation[3]

        # 将 coordinates 转换为边界框（裁剪区域）
        x_coords = coordinates[0::2]  # 提取 x 坐标
        y_coords = coordinates[1::2]  # 提取 y 坐标
        left, right = min(x_coords), max(x_coords)
        top, bottom = min(y_coords), max(y_coords)

        # 裁剪文字区域
        cropped_image = original_image.crop((left, top, right, bottom))

        # 保存裁剪后的文字区域图片到 images 文件夹中
        cropped_image_filename = f'image_{image_id}_region_{i}.{image_format}'
        cropped_image_path = os.path.join(dataset.path, 'images', cropped_image_filename)
        cropped_image.save(cropped_image_path)

        # 将标注结果存储到数据库
        new_data = Data(
            dataset_id=dataset.id,
            image_path=cropped_image_path,  # 保存裁剪图片的路径
            text=text,
            is_handwritten=is_handwritten,
            coordinates=','.join(map(str, coordinates)),  # 将坐标数组转换为字符串存储
            confidence=confidence
        )
        db.session.add(new_data)

    db.session.commit()

    # 返回存储后的数据供前端显示
    response_data = Data.query.filter_by(dataset_id=dataset.id).all()
    response = [{
        "id": d.id,
        "image_path": d.image_path,
        "text": d.text,
        "is_handwritten": d.is_handwritten,
        "coordinates": d.coordinates,
        "confidence": d.confidence
    } for d in response_data]

    return jsonify(response), 200


# 获取某个数据集的标注信息
@datasets_api.route('/annotations/<dataset_name>', methods=['GET'])
def get_annotations(dataset_name):
    dataset = Dataset.query.filter_by(name = dataset_name).first()
    dataset_id = dataset.id
    data_entries = Data.query.filter_by(dataset_id=dataset_id).all()
    response = [{
        "image_path": d.image_path,
        "annotation": d.text,
        "created_at": d.created_at
    } for d in data_entries]
    return jsonify(response), 200
@datasets_api.route('/annotations_filter_by_imagename/', methods=['post'])
def get_annotations_filter_by_imagename():
    data = request.json
    image_name = data['imagename']
    image_name = image_name.split('.')[0]
    dataset = Dataset.query.filter_by(name = data['datasetName']).first()
    dataset_id = dataset.id
    data_entries = Data.query.filter_by(dataset_id=dataset_id).all()
    response = [{
        "image_path": d.image_path.split('/')[-1],
        "annotation": d.text,
        "confidence": d.confidence
    } for d in data_entries]
    result = []
    for i in response:
        if image_name[9:] in i['image_path']:
            result.append(i)
    return jsonify(result), 200


@datasets_api.route('/delete_annotations', methods=['POST'])
def delete_annotations():
    data = request.get_json()
    ids_to_delete = data.get('ids')  # 前端传来的需要删除的ID列表

    for data_id in ids_to_delete:
        data_entry = Data.query.get(data_id)
        if data_entry:
            # 删除裁剪后的文字区域图片
            if os.path.exists(data_entry.image_path):
                os.remove(data_entry.image_path)

            # 删除数据库记录
            db.session.delete(data_entry)

    db.session.commit()

    return jsonify({"message": "删除成功"}), 200

@datasets_api.route('/datasets/delete_image/<dataset_name>', methods=['POST'])
def delete_image(dataset_name):
    data = request.get_json()
    image_id = data.get('imageId')

    if not image_id:
        return jsonify({"error": "Missing imageId"}), 400

    # 构建图片文件的路径（这里假设图片存储在特定的文件夹中）
    image_folder = os.path.join("./datasets", dataset_name)
    image_path = os.path.join(image_folder, f'image_{image_id}.png')

    # 删除图片文件
    if os.path.exists(image_path):
        os.remove(image_path)
    else:
        return jsonify({"error": "Image not found"}), 404

    # 删除相关的标注数据（假设标注数据存储在数据库或文本文件中）
    annotations_file = os.path.join("./datasets", dataset_name, 'annotations.txt')

    # 示例：从文本文件中删除与该图片相关的标注条目
    with open(annotations_file, 'r') as file:
        lines = file.readlines()
    with open(annotations_file, 'w') as file:
        for line in lines:
            if f'image_{image_id}' not in line:
                file.write(line)

    return jsonify({"success": True}), 200