import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import numpy as np
from resunet import U_Net
from steg import CM_UNet
from charformer import CharFormer
from AST import AST
# from swinir import SwinIR
# from APBSN import APBSN
import time

def test(img_path, savepath, G):
    # 读取图像（彩色）
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return 0  # 返回零时间，表示跳过此图像

    print(f"原始图像形状: {img.shape}")

    # 定义转换
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ])

    # 应用转换
    img_tensor = transform_pipeline(img.copy())  # [C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to('cuda')  # [1, C, 128, 128]

    print(f"转换后图像形状: {img_tensor.shape}")

    # 设置 GPU 时间记录
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 推理阶段：开始计时
    start_event.record()

    with torch.no_grad():
        fake_img = G(img_tensor)

    # 推理阶段：结束计时
    end_event.record()

    # 等待 GPU 完成计算
    torch.cuda.synchronize()

    # 计算推理时间（单位：毫秒）
    inference_time_ms = start_event.elapsed_time(end_event)  # 毫秒
    inference_time_sec = inference_time_ms / 1000  # 转换为秒
    print(f"Inference Time for {img_path}: {inference_time_sec:.6f} seconds")

    # 应用 IWT
    # fake_img = iwt(dwt_fake_img)

    print(f"生成图像形状: {fake_img.shape}")

    # 限制值在 [0, 1] 之间
    fake_img = torch.clamp(fake_img, 0, 1)

    # 转换为 NumPy 数组
    fake_img = fake_img.cpu().detach().numpy() * 255  # [1, C, H, W] * 255

    # 去除批次维度
    fake_img = np.squeeze(fake_img, axis=0)  # [C, H, W]

    # 转置为 [H, W, C]
    fake_img = np.transpose(fake_img, (1, 2, 0))  # [H, W, C]

    print(f"转换后 NumPy 图像形状: {fake_img.shape}")

    # 如果图像是单通道，去除通道维度
    if fake_img.shape[2] == 1:
        fake_img = fake_img.squeeze(2)  # [H, W]

    # 转换为 uint8
    fake_img = fake_img.astype(np.uint8)

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    # 保存图像为 PNG 格式
    success = cv2.imwrite(savepath, fake_img)
    if not success:
        print(f"保存图像失败: {savepath}")
    else:
        print(f"图像已保存为 PNG 格式: {savepath}")

    return inference_time_sec  # 返回推理时间

def main():
    total_time = 0
    num_tests = 0
    #G = U_Net(img_ch=3,output_ch=3).to('cuda')
    G = AST(img_size=128, in_chans=3, dd_in=3).to('cuda')
    #G = CM_UNet(in_nc=3).to('cuda')
    # 实例化网络
    # upscale = 1  # 对于去噪任务，通常设置为 1
    # window_size = 8  # 根据需要设置窗口大小
    # height, width = 128, 128  # 假设的输入图像尺寸
    # in_chans = 3  # 输入通道数（RGB 图像）
    # img_range = 1.0
    # G = SwinIR(
    #     img_size=(height, width),  # 图像尺寸
    #     in_chans=in_chans,  # 输入通道数
    #     upscale=upscale,  # 去噪任务使用 1
    #     window_size=window_size,  # 窗口大小
    #     img_range=img_range,  # 图像范围
    #     embed_dim=60,  # 嵌入维度
    #     depths=[6, 6, 6, 6],  # 每层深度
    #     num_heads=[6, 6, 6, 6],  # 每层的注意力头数
    #     mlp_ratio=2,  # MLP 比例
    #     upsampler='nearest+conv',  # 去噪任务使用 'nearest+conv'
    #     resi_connection='1conv'  # 默认残差连接方式
    # ).to('cuda')
    # G = CharFormer(
    #      dim=64,  # initial dimensions after input projection
    #     stages=3,  # number of stages
    #     depth_RSAB=2,  # number of transformer blocks per RSAB
    #     depth_GSNB=1,  # number of Conv2d blocks per GSNB
    #     window_size=8,  # window size for attention
    #     dim_head=32,  # dimension of attention heads
    #     heads=2,  # number of attention heads
    # ).to('cuda')

    # 加载预训练权重
    ckpt_path = '/home/u2308283094/wangyu/pix2pix/wz/checkpoint/Hmodel.pth'
    if not os.path.exists(ckpt_path):
        print(f"预训练权重文件不存在: {ckpt_path}")
    else:
        G.load_state_dict(torch.load(ckpt_path))
        G.eval()

        # 数据路径
        datapath = '/home/u2308283094/wangyu/pix2pix/testreal'
        savepath_root = '/home/u2308283094/wangyu/pix2pix/scunetreal'

        if not os.path.exists(datapath):
            print(f"数据路径不存在: {datapath}")
        else:
            # 获取文件列表
            file_list = os.listdir(datapath)
            for name in file_list:
                path = os.path.join(datapath, name)
                
                # 构建保存路径，确保扩展名为 .png
                base_name = os.path.splitext(name)[0]
                save = os.path.join(savepath_root, base_name + '.png')
                
                # 调用测试函数，计算每次推理时间
                inference_time = test(path, save, G)
                total_time += inference_time
                num_tests += 1

            # 计算并打印平均推理时间
            if num_tests > 0:
                avg_inference_time = total_time / num_tests
                print(f"平均推理时间: {avg_inference_time:.6f} 秒")
            else:
                print("没有测试图像.")

if __name__ == '__main__':
    main()
