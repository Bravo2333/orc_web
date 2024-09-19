import base64

import cv2
import numpy as np
import matplotlib.pyplot as plt
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class CCNet:
    def __init__(self):
        self.table_recognition = pipeline(Tasks.table_recognition,
                                          model='damo/cv_dla34_table-structure-recognition_cycle-centernet')

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
