from flask import Flask, request, jsonify, send_from_directory
import os
import base64
from TSR import CCNet

ccnet = CCNet()
app = Flask(__name__)

# 设置上传文件的静态目录
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 模拟表格识别处理函数
def process_image(file_path):
    # 模拟表格识别后的base64图片（将原图转换为base64）
    encoded_string ,polygons= ccnet.table_recognition(file_path)


    # 模拟的识别结果文本
    recognition_result = "这是模拟的表格识别结果。"

    return {
        "image_base64": encoded_string,
        "result": recognition_result
    }


# API 接口: 处理图像识别请求
@app.route('/api/recognize', methods=['POST'])
def recognize_table():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # 保存上传的图片
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # 处理图片，模拟表格识别并返回结果
        result = process_image(file_path)

        # 返回识别的图片和文本结果
        return jsonify({
            "image": {
                "base64": f"data:image/jpeg;base64,{result['image_base64']}"
            },
            "result": result["result"]
        })

    return jsonify({"error": "Something went wrong"}), 500


# 提供静态文件（图片）
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
