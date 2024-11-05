import base64
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from multiprocessing import Process, Manager

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
# 将图像数据转化为 PyTorch 张量，并将其加载到 GPU
def image_to_gpu_tensor(image):
    image = image.convert('RGB')
    image_np = np.array(image)
    # 使用 PyTorch 将图像数据加载到 GPU
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image_tensor


# 通过 GPU 执行多边形裁剪
def crop_polygon_with_gpu(image_path, annotations, result_dict, device):
    num = 0
    image = Image.open(image_path)
    image_tensor = image_to_gpu_tensor(image).to(device)  # 将图像数据加载到 GPU

    for annotation in annotations:
        polygon = Polygon(annotation['points'])

        # 创建掩码图像
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon.exterior.coords, fill=255)

        # 将掩码加载到 GPU
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.float32).to(device) / 255.0

        # 使用掩码提取多边形区域
        cropped_image_tensor = image_tensor * mask_tensor.unsqueeze(0)

        # 将裁剪结果转回 CPU
        cropped_image_np = cropped_image_tensor.permute(1, 2, 0).cpu().numpy()

        # 保存裁剪图像
        temp = image_path.split('/')[-1].split('.')
        img_filename = f"{temp[0]}_{num}.{temp[1]}"

        save_path = os.path.join("annotation_Dataset_rec", image_path.split('/')[1])
        os.makedirs(save_path, exist_ok=True)

        cropped_image = Image.fromarray((cropped_image_np * 255).astype(np.uint8))
        cropped_image.save(os.path.join(save_path, img_filename), format='JPEG')

        num += 1

    result_dict[image_path] = f"{len(annotations)} images processed."


# 构造处理函数的参数
def construct_parameters_for_processing(image_paths, annotations):
    params = []
    for image_path, annotation in zip(image_paths, annotations):
        # 构造参数，每次都准备下一张图的参数
        params.append((image_path, annotation))
    return params


# 启动进程并处理
def start_processing_with_multiprocessing(params):
    manager = Manager()
    result_dict = manager.dict()  # 用于存储每个进程的返回值)

    processes = []
    for param in params:
        # 创建新进程，传入每一组参数
        process = Process(target=crop_polygon_with_gpu, args=(param[0], param[1], result_dict,"cuda"))
        processes.append(process)
        process.start()  # 启动进程

    # 等待所有进程完成
    for process in processes:
        process.join()

    # 输出每个进程的结果
    for image_path, result in result_dict.items():
        print(f"Result for {image_path}: {result}")