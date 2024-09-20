import requests
import base64
import json

# Flask服务器地址
url = "http://127.0.0.1:5000/api/recognize"  # 替换为你的Flask接口地址

# 本地待上传图片路径
image_path = "table1_1.png"
output_image_path = "output_image.jpg"
output_text_path = "result.txt"

# 打开图片文件，以二进制形式上传
with open(image_path, "rb") as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

# 检查响应状态
if response.status_code == 200:
    # 获取服务器返回的json数据
    response_data = response.json()

    # 从返回数据中提取base64编码的图片
    base64_image = response_data['image']['base64']

    # 解码base64并保存为图片文件
    with open(output_image_path, "wb") as output_image:
        output_image.write(base64.b64decode(base64_image))

    # 提取识别结果并保存到文本文件中
    recognition_results = response_data['result']
    with open(output_text_path, "w") as output_text:
        output_text.write(recognition_results)

    print(f"Image saved to {output_image_path}")
    print(f"Recognition results saved to {output_text_path}")
else:
    print(f"Failed to upload image. Status code: {response.status_code}")
