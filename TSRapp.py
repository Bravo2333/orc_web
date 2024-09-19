from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class CCNet:
    def __init__(self):
        self.table_recognition = pipeline(Tasks.table_recognition,model='damo/cv_dla34_table-structure-recognition_cycle-centernet')

    def do_TSR(self, filepath):
        result = self.table_recognition(filepath)
        image = cv2.imread(filepath)
        # 定义多边形的坐标
        polygons = result['polygons']
        # 遍历每个多边形并在图像上绘制
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            # 在图片上绘制多边形，颜色为蓝色，线条宽度为2
            cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
        # 将BGR格式的图片转换为RGB格式，以便使用matplotlib显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', image_rgb)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        # 使用matplotlib显示图片
        plt.imshow(image_rgb)
        plt.axis('off')  # 不显示坐标轴
        plt.show()
        return image_base64,polygons
app = Flask(__name__)
ccnet = CCNet()
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image
def filter_polygons_by_area(polygons: np.ndarray) -> np.ndarray:
    """
    按面积大小排序并删除面积小于后20%中面积最大的多边形面积的80%的多边形
    :param polygons: 输入的多边形列表, 形如 [N, M, 2] 的 ndarray，N 是多边形个数，M 是顶点数，2 表示 x, y 坐标
    :return: 过滤后的多边形列表，格式同输入
    """
    if polygons.size == 0:
        return np.array([])

    # Step 1: 计算每个多边形的面积
    polygon_areas = [(i, cv2.contourArea(polygons[i])) for i in range(polygons.shape[0])]

    # Step 2: 按面积大小排序
    polygon_areas.sort(key=lambda x: x[1])

    # Step 3: 计算后20%中面积最大的多边形的80%的面积
    num_polygons = len(polygon_areas)
    last_20_percent_count = num_polygons // 5  # 取后20%的多边形数量
    if last_20_percent_count == 0:
        last_20_percent_count = 1  # 至少取一个多边形
    last_20_percent_polygons = polygon_areas[-last_20_percent_count:]

    # 找到后20%多边形中最大面积的多边形
    max_area_in_last_20 = max([area for _, area in last_20_percent_polygons])

    # 计算最大面积的80%
    threshold_area = 0.8 * max_area_in_last_20

    # Step 4: 删除面积小于阈值的多边形，返回过滤后的多边形
    filtered_indices = [i for i, area in polygon_areas if area >= threshold_area]
    filtered_polygons = polygons[filtered_indices]

    return filtered_polygons
# API 接口: 处理图像识别请求
@app.route('/api/recognize', methods=['POST'])
def recognize_table():
    data = request.json
    base64_image = data.get('image')

    if not base64_image:
        return jsonify({"error": "No image provided"}), 400

    # 将Base64编码的图片转换为OpenCV图像
    image = base64_to_image(base64_image)
    cv2.imwrite('temp.jpg',image)
    encoded_string, polygons = ccnet.do_TSR('temp.jpg')
    print(polygons)
    polygons = filter_polygons_by_area(polygons)
    result = {
        "image_base64": encoded_string,
        "polygons": polygons.tolist()
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True,port=3000)
