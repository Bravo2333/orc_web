import paddle.inference as paddle_infer
import numpy as np


def infer_with_paddle(pdmodel_file, pdiparams_file, input_data):
    # 创建 config 并设置模型文件和参数文件
    config = paddle_infer.Config(pdmodel_file, pdiparams_file)

    # 设置使用 CPU 或 GPU，假设这里使用 CPU
    config.disable_gpu()  # 如果需要 GPU，使用 config.enable_use_gpu(gpu_memory_mb, gpu_id)

    # 设置 precision mode（可以根据需要选择 FP32, FP16, INT8 等）
    config.switch_ir_optim(True)  # 打开IR优化

    # 创建预测器
    predictor = paddle_infer.create_predictor(config)

    # 获取输入句柄
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])  # 假设模型有一个输入

    # 设置输入数据
    input_handle.copy_from_cpu(input_data)

    # 运行推理
    predictor.run()

    # 获取输出句柄
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])

    # 获取输出数据
    output_data = output_handle.copy_to_cpu()

    return output_data


# 示例调用
if __name__ == "__main__":
    pdmodel_file = './text_detection_module/best_accuracy.pdmodel'
    pdiparams_file = './text_detection_module/best_accuracy.pdiparams'

    # 构造输入数据（假设输入是形状为 [1, 3, 224, 224] 的图像）
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # 调用推理函数
    output_data = infer_with_paddle(pdmodel_file, pdiparams_file, input_data)

    # 打印输出
    print("Inference Output:", output_data)
