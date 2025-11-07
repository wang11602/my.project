import os
import csv
# os.environ["CUDA_VISIBLE_DEVICES"]='1'  # 指定使用1号显卡为主卡
import torch
import torch.nn
from torch import nn
import torch.optim
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import structural_similarity as ssim
import wz_config as c
import random
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
import warnings
import matplotlib.pyplot as plt
import logging
import timm
import timm.scheduler
from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from math import exp
from resunet import U_Net
from cmunet import CM_UNet
from stega import CM_UNet
from charformer import CharFormer
from AST import AST
# from swinir import SwinIR
# from model1 import SCUNet
# from APBSN import APBSN
# from wzl_model import stegDcnv4
from mydatasets import CreateDatasets
from split_data import split_data
# from stegtrans import stegDcnv4
from wz_critic import *
# from wz_common import DWT,IWT

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2024)  # 将42替换为您喜欢的任何种子值


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, eps=1e-6):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # 权重系数
        self.eps = eps  # Charbonnier Loss 平滑项

    def forward(self, X, Y):
        # L2 损失
        l2_loss = torch.mean((X - Y) ** 2)
        
        # Charbonnier 损失
        diff = X - Y
        charbonnier_loss = torch.mean(torch.sqrt(diff ** 2 + self.eps))
        
        # 组合损失
        combined_loss = self.alpha * l2_loss + (1 - self.alpha) * charbonnier_loss
        return combined_loss
class CannyEdgeLoss(nn.Module):
    def __init__(self, low_threshold=50, high_threshold=150):
        super(CannyEdgeLoss, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, pred, target):
        # 将 Tensor 转为 NumPy
        pred_np = pred.squeeze().detach().cpu().numpy() * 255  # 假设输入范围为[0, 1]
        target_np = target.squeeze().detach().cpu().numpy() * 255

        # 使用 OpenCV 提取边缘
        pred_edges = cv2.Canny(pred_np.astype(np.uint8), self.low_threshold, self.high_threshold)
        target_edges = cv2.Canny(target_np.astype(np.uint8), self.low_threshold, self.high_threshold)

        # 转回 Tensor
        pred_edges_tensor = torch.from_numpy(pred_edges).float().to(pred.device) / 255.0
        target_edges_tensor = torch.from_numpy(target_edges).float().to(target.device) / 255.0

        # 计算 L1 损失
        loss = torch.mean(torch.abs(pred_edges_tensor - target_edges_tensor))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel算子定义，适配三通道图像
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, X, Y):
        # 将Sobel算子拓展到适配3通道图像
        sobel_x = self.sobel_x.repeat(3, 1, 1, 1).to(X.device)  # [3, 1, 3, 3]
        sobel_y = self.sobel_y.repeat(3, 1, 1, 1).to(X.device)  # [3, 1, 3, 3]

        # 使用分组卷积对每个通道独立应用Sobel算子
        X_edge_x = F.conv2d(X, sobel_x, padding=1, groups=3)
        X_edge_y = F.conv2d(X, sobel_y, padding=1, groups=3)
        X_edges = torch.sqrt(X_edge_x ** 2 + X_edge_y ** 2 + 1e-6)

        Y_edge_x = F.conv2d(Y, sobel_x, padding=1, groups=3)
        Y_edge_y = F.conv2d(Y, sobel_y, padding=1, groups=3)
        Y_edges = torch.sqrt(Y_edge_x ** 2 + Y_edge_y ** 2 + 1e-6)

        # 计算L1损失
        loss = F.l1_loss(X_edges, Y_edges)
        return loss



def computePSNR(origin, pred):
    img_1 = np.array(origin).astype(np.float64)*255
    img_2 = np.array(pred).astype(np.float64)*255
    mse = np.mean((img_1 - img_2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def setup_logging():
    # 配置日志记录
    logging.basicConfig(level=logging.INFO, filename=c.log_path, filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

def main():
    warnings.filterwarnings("ignore")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_logging()
    writer = SummaryWriter(logdir=c.t_log, comment='tt', filename_suffix="steg")
    Hnet = CM_UNet(in_nc=3)
    # Hnet = APBSN(in_ch=3)
    # Hnet = stegDcnv4(in_channels=3, out_channels=3)
    Hnet.to(device)

    if c.is_load:
        Hnet.load_state_dict(torch.load(c.Hload))


    Hoptim = torch.optim.AdamW(Hnet.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)


    Hscheduler = timm.scheduler.CosineLRScheduler(optimizer=Hoptim, t_initial=c.epochs, lr_min=1e-7, warmup_t=0,
                                                  warmup_lr_init=5e-6)

    train_imglist, val_imglist = split_data(c.data_path)
    train_datasets = CreateDatasets(train_imglist, c.img_size)
    val_datasets = CreateDatasets(val_imglist, c.img_size)

    train_loader = DataLoader(dataset=train_datasets, batch_size=c.batch, shuffle=True, num_workers=8,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_datasets, batch_size=c.batch, shuffle=True, num_workers=8,
                            drop_last=True)
    ELoss = CannyEdgeLoss().to(device)
    loss = L1_Charbonnier_loss().to(device)
    loss1 = nn.L1Loss().to(device)
    loss2 = CombinedLoss().to(device)
    best_loss = float('inf')
    batch_idx = 0
    val_loss_list = []


    for epoch in range(c.epochs):
        loop = tqdm((train_loader),total = len(train_loader),leave=True)
        Hnet.train()
        
        total_loss = 0.0

        for data in loop:
            in_img = data[0].to(device)

            # dwt_in_img=dwt(in_img)

            real_img = data[1].to(device)
            # dwt_real_img = dwt(real_img)

            Hoptim.zero_grad()
        
            H_output = Hnet(in_img)
            # iwt_H_output = iwt(H_output) 
            
            # Hloss = loss(H_output, real_img) + ELoss(H_output, real_img)
            Hloss = loss1(H_output, real_img) 
            All_loss = Hloss 

            # 使用scaler缩放损失并执行反向传播
            All_loss.backward()

            # 使用scaler执行优化器的更新步骤
            Hoptim.step()

            total_loss += All_loss.item()
            loop.set_description(f'Train Epoch [{epoch}/{c.epochs}]')
            loop.set_postfix({'Hloss': Hloss.item()})
            batch_idx += 1

        # 学习率调度和日志记录
        current_lr = Hoptim.param_groups[0]['lr']
        logging.info(
            f'Train Epoch [{epoch}/{c.epochs}] All_loss: {total_loss / len(train_loader)} HCurrent_lr: {current_lr}')
        writer.add_scalars("Train", {"Train_Loss": total_loss / len(train_loader)}, epoch + 1)

        # 验证循环
        loop = tqdm((val_loader), total=len(val_loader), leave=True)
        Hnet.eval()

        
        with torch.no_grad():
            psnr_c = []
            ssim_c = []
            y_psnr_c = []        
            total_loss = 0.0
            for data in loop:
                in_img = data[0].to(device)

                # dwt_in_img=dwt(in_img)

                real_img = data[1].to(device)
                # dwt_real_img = dwt(real_img)

                Hoptim.zero_grad()
            
                H_output = Hnet(in_img)
                # iwt_H_output = iwt(H_output) 
                
                # Hloss = loss(H_output, real_img) + ELoss(H_output, real_img)
                Hloss = loss1(H_output, real_img)
                total_loss += Hloss

                cover = H_output.clamp(0, 1).cpu().numpy()
                real_img = real_img.cpu().numpy()
                
                for i in range(cover.shape[0]):
                    # 灰度图只取第0通道
                    pred_img = cover[i][0]   # shape [H,W]
                    gt_img   = real_img[i][0]

                    # 转 uint8
                    pred_img_uint8 = (pred_img * 255).astype(np.uint8)
                    gt_img_uint8   = (gt_img * 255).astype(np.uint8)

                    # 计算 PSNR/SSIM
                    psnr_val = sk_psnr(gt_img_uint8, pred_img_uint8, data_range=255)
                    ssim_val = sk_ssim(gt_img_uint8, pred_img_uint8, data_range=255)

                    psnr_c.append(psnr_val)
                    ssim_c.append(ssim_val)


                loop.set_description(f'Val')
                loop.set_postfix({'Hloss': Hloss.item()})

            # 每轮平均
            avg_psnr = np.mean(psnr_c)
            avg_ssim = np.mean(ssim_c)
            val_loss = total_loss / len(val_loader)

            logging.info(
                f'Val Epoch [{epoch}/{c.epochs}] val_loss: {val_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}'
            )
            writer.add_scalars("Validation", {
                "val_Loss": val_loss,
                "PSNR": avg_psnr,
                "SSIM": avg_ssim
            }, epoch + 1)

        

            # === 新增：存储每轮验证结果 ===
            val_loss_list.append(val_loss.item() if torch.is_tensor(val_loss) else val_loss)



            # 保存到CSV文件（修改这里的变量名）
        with open('val_loss_only.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)  # 将 writer 改为 csv_writer
            csv_writer.writerow(['Epoch', 'Validation_Loss'])  # 这里也要改
            for epoch_idx, loss_val in enumerate(val_loss_list):
                csv_writer.writerow([epoch_idx + 1, loss_val])  # 这里也要改

            Hscheduler.step(epoch)

            # 保存最佳模型
            if best_loss > total_loss:
                best_loss = total_loss
                torch.save(Hnet.state_dict(), f'{c.HMODEL_PATH}/Hmodel.pth')
                total_loss = 0.0
                logging.info(f'model checkpoint saved!')

            # 每100个epoch保存一次模型
            if epoch % 100 == 0:
                torch.save(Hnet.state_dict(), f'{c.HMODEL_PATH_100}/Hmodel.pth')
                logging.info(f'model checkpoint saved!')

# === 新增：训练结束后画图 ===
    epochs = range(1, len(val_loss_list) + 1)

    # 验证损失曲线
    plt.figure(figsize=(6,4))
    plt.plot(epochs, val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_loss_curve.png", dpi=300)
    plt.show()

    # PSNR曲线
    plt.figure(figsize=(6,4))
    plt.plot(epochs, psnr_list, label="PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("psnr_curve.png", dpi=300)
    plt.show()

    # SSIM曲线
    plt.figure(figsize=(6,4))
    plt.plot(epochs, ssim_list, label="SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("ssim_curve.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()


