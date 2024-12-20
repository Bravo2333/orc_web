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
def crop_polygon_with_gpu(image_path, annotations, result_dict, progress_queue, device):
    image = Image.open(image_path)

    total_annotations = len(annotations)
    for index, annotation in enumerate(annotations):
        polygon = Polygon(annotation['points'])

        # 获取多边形的最小边界框
        x_min, y_min, x_max, y_max = polygon.bounds
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # 裁剪图像的边界框区域
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_tensor = image_to_gpu_tensor(cropped_image).to(device)

        # 创建掩码图像并裁剪到同样的边界框
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon.exterior.coords, fill=255)
        mask_cropped = mask.crop((x_min, y_min, x_max, y_max))
        mask_tensor = torch.tensor(np.array(mask_cropped), dtype=torch.float32).to(device) / 255.0

        # 使用掩码提取多边形区域
        final_cropped_tensor = cropped_tensor * mask_tensor.unsqueeze(0)

        # 将裁剪结果转回 CPU
        final_cropped_np = final_cropped_tensor.permute(1, 2, 0).cpu().numpy()

        # 设置背景为白色（或透明）
        final_cropped_np[mask_tensor.cpu().numpy() == 0] = 255  # 白色背景

        # 保存仅包含多边形区域的裁剪图像
        temp = image_path.split('/')[-1].split('.')
        img_filename = f"{temp[0]}_{index}.{temp[1]}"

        save_path = os.path.join("annotation_Dataset_rec",image_path.split('/')[1],"images")
        os.makedirs(save_path, exist_ok=True)

        final_image = Image.fromarray((final_cropped_np * 255).astype(np.uint8))
        final_image.save(os.path.join(save_path, img_filename), format='JPEG')

        # 更新进度信息
        progress_queue.put((image_path, index + 1, total_annotations))

    result_dict[image_path] = f"{len(annotations)} images processed."


# 构造处理函数的参数
def construct_parameters_for_processing(image_paths, annotations):
    params = []
    for image_path, annotation in zip(image_paths, annotations):
        # 构造参数，每次都准备下一张图的参数
        params.append((image_path, annotation))
    return params


# 显示进度
def display_progress(progress_queue):
    while True:
        try:
            # 获取进度信息
            image_path, completed, total = progress_queue.get(timeout=1)
            progress = (completed / total) * 100
            print(f"Processing {image_path}: {completed}/{total} ({progress:.2f}%) completed.")
        except:
            # 如果队列为空，跳出循环
            break


# 启动进程并处理
def start_processing_with_multiprocessing(params):
    manager = Manager()
    result_dict = manager.dict()  # 用于存储每个进程的返回值
    progress_queue = manager.Queue()  # 用于存储进度更新

    # 拆分为批次，每个批次包含最多 batch_size 个任务
    batches = [params[i:i + 100] for i in range(0, len(params), 100)]

    for batch in batches:
        processes = []

        # 启动进度显示线程
        progress_thread = Process(target=display_progress, args=(progress_queue,))
        progress_thread.start()

        for param in batch:
            # 创建新进程，传入每一组参数
            process = Process(target=crop_polygon_with_gpu,
                              args=(param[0], param[1], result_dict, progress_queue, "cpu"))
            processes.append(process)
            process.start()  # 启动进程

        # 等待当前批次的所有进程完成
        for process in processes:
            process.join()

        # 结束进度显示线程
        progress_thread.terminate()

        # 输出每个进程的结果
        for image_path, result in result_dict.items():
            print(f"Result for {image_path}: {result}")