from det_infer import det
from rec_infer import rec
import os
import json

import cv2
# 从推理结果文件中逐行读取并解析推理数据
def load_inference_results(file_path):
    inference_results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            img_path, annotations = line.strip().split('\t')
            annotations = json.loads(annotations)
            inference_results[img_path] = annotations
    return inference_results


# 加载train.txt或val.txt的图像路径和标注
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            img_path, annotations = line.strip().split('\t')
            # annotations = json.loads(annotations)
            data.append((img_path, annotations))
    return data


# 保存处理后的数据
def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for img_path, annotations in data:
            line = f"{img_path}\t{json.dumps(annotations, ensure_ascii=False)}\n"
            f.write(line)


# 根据推理结果切割图像并保存裁剪后的图片
def process_images_with_inference(data, inference_results, output_rec_dir):
    processed_data = []

    if not os.path.exists(output_rec_dir):
        os.makedirs(output_rec_dir)

    for img_path, annotations in data:
        image = cv2.imread('../dataset/g_test/'+img_path)
        if image is None:
            print(f"Error: Unable to read image {img_path}")
            continue

        img_name = os.path.basename(img_path)
        inference_result = inference_results.get('../dataset/g_test\\'+img_path)

        if inference_result is None:
            print(f"No inference result for {img_path}")
            continue

        for idx, ann in enumerate(inference_result):
            points = ann['points']
            # transcription = ann['transcription']

            # 获取四个点的坐标
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # 裁剪图像
            cropped_image = image[y_min:y_max, x_min:x_max]

            # 保存裁剪后的图像
            rec_img_name = f"{os.path.splitext(img_name)[0]}_rec_{idx}.jpg"
            rec_img_path = os.path.join(output_rec_dir, rec_img_name)
            cv2.imwrite(rec_img_path, cropped_image)

            # 更新annotation中的图片路径为裁剪后的图像路径
            ann['rec_image'] = rec_img_path

        # 将处理后的数据保存
        processed_data.append((img_path, inference_result))

    return processed_data
filelist = []
dataset_dir = '../dataset/g_test'  # 假设数据集文件夹
output_rec_dir = './afterdet'
inference_results_path = 'det_result.txt'  # 推理结果文件路径
# train_data = load_data(os.path.join(dataset_dir, 'train.txt'))
# val_data = load_data(os.path.join(dataset_dir, 'val.txt'))
# for i in train_data:
#     filelist.append(os.path.join(dataset_dir, i[0]))
# for i in val_data:
#     filelist.append(os.path.join(dataset_dir, i[0]))
# print("开始检测")
detinstence = det()
detinstence.det(detinstence.config, detinstence.logger, filelist)
print("结束检测，开始切割")

inference_results = load_inference_results(inference_results_path)
processed_train_data = process_images_with_inference(train_data, inference_results, output_rec_dir)
processed_val_data = process_images_with_inference(val_data, inference_results, output_rec_dir)

images = os.listdir('./afterdet')
torec = []
for i in images:
    torec.append('./afterdet/'+i)
print(1234)
recinstence = rec()
recinstence.rec(recinstence.config,recinstence.logger,torec)