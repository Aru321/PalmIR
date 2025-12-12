import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import datetime
from pathlib import Path
import logging
import json
import numpy as np
from torch.xpu import device

# exp configs
from configs.configs_plain_unet import get_config_unet_small, get_config_unet
from configs.configs_plain_DHA import get_config_DHA
from configs.configs_plain_GUNet import get_config_gunet
from configs.configs_PalmIR import get_config_palmir_ablation, get_config_palmir_base
from configs.configs_plain_AdaIR import get_config_adair_base

from utils.EMA import EMA
from utils.metrics import calculate_psnr,calculate_nmse,calculate_ssim


class PalmprintTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 设置随机种子
        self.set_seed(config.get('seed', 42))

        # 初始化组件
        self.model = None
        self.model_name = None
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.scaler = 10

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_psnr = 0.0
        self.best_nmse = 1.0
        self.best_ssim = 0.0

        # 创建输出目录
        self.setup_directories()

        # 设置日志
        self.setup_logging()

        # 记录配置
        self.logger.info(f"Training configuration: {json.dumps(config, indent=2)}")

    def set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_directories(self):
        """创建输出目录"""
        self.output_dir = Path(self.config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'

        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

    def setup_logging(self):
        """设置日志"""
        log_file = self.log_dir / f"training_{self.config['model']['name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # 配置logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def build_model(self):
        """构建模型"""
        model_config = self.config['model']
        model_name = model_config['name']

        # 这里根据模型名称创建模型，您需要根据实际情况实现
        if model_name == 'PalmIR':
            from models.PalmIR import PalmIR
            self.model = PalmIR(channels=[64,128,256,512],device=self.device)
            self.model_name = model_name

        elif model_name == 'PalmIR_Ablation':
            from models.PalmIR import PalmIR_Ablation
            self.model = PalmIR_Ablation(channels=[64,128,256,512])
            self.model_name = model_name

        elif model_name == 'UNet_Small':
            from models.UNet import UNetSmall
            self.model = UNetSmall(num_channels=3,num_classes=3)
            self.model_name = model_name

        elif model_name == 'UNet':
            from models.UNet import UNet
            self.model = UNet(num_channels=3,num_classes=3)
            self.model_name = model_name

        elif model_name == 'GUNet':
            from models.GUNet import GUNet
            self.model = GUNet(3,3,64)
            self.model_name = model_name

        elif model_name == 'VIRNet':
            from models.VIRNet import VIRAttResUNet
            self.model = VIRAttResUNet(im_chn=3)
            self.model_name = model_name

        elif model_name == 'SOTA_DHA':
            from models.DHA_Compare import DHA
            self.model = DHA(
                in_channels=3,
                out_channels=3,
                base_dim=64,
                num_rhabs=self.config['num_blocks'],
                num_heads=4,
                window_size=8,
                input_size=self.config['input_size'],
            )
            self.model_name = model_name

        elif model_name == 'AdaIR':
            from models.AdaIR_Compare import AdaIR
            self.model = AdaIR()
            self.model_name = model_name

        else:
            raise NotImplementedError

        self.model = self.model.to(self.device)
        self.logger.info(f"Model built: {model_name}")
        self.logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def build_optimizer(self):
        """构建优化器"""
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name']
        lr = optimizer_config['lr']

        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        elif optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # 学习率调度器
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('name') == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 60),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config.get('name') == 'CosineAnnealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.config['training']['epochs'])
            )

        # EMA
        if self.config.get('use_ema', True):
            self.ema = EMA(self.model, decay=self.config.get('ema_decay', 0.995))

        self.logger.info(f"Optimizer built: {optimizer_name}")

    def build_criterion(self):
        """构建损失函数"""
        criterion_config = self.config['criterion']
        criterion_name = criterion_config['name']

        if criterion_name == 'L1':
            self.criterion = nn.L1Loss()
        elif criterion_name == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion_name == 'SmoothL1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")
        self.scaler = 10
        self.logger.info(f"Criterion built: {criterion_name}")

    def build_dataloaders(self):
        """构建数据加载器"""
        from datautils import DegradationPalmprintDataset  # 替换为您的数据集类
        from datautils_test import TestDegradationDataset
        data_config = self.config['data']

        # 训练集
        train_dataset = DegradationPalmprintDataset(
            data_folder=data_config['train_path'],
            transform=self.get_transforms('train'),
            num_ref_samples=data_config.get('num_ref_samples', 8),
            apply_degradation=True,
            apply_mode=self.config['data']['apply_mode'],
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['train_batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        # 验证集
        val_dataset = TestDegradationDataset(
            data_folder=data_config['val_path'],
            transform=self.get_transforms('val'),
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['val_batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )

        self.logger.info(f"Train dataset: {len(train_dataset)} samples")
        self.logger.info(f"Val dataset: {len(val_dataset)} samples")

    def get_transforms(self, phase='train'):
        """获取数据变换"""
        from torchvision import transforms

        if phase == 'train':
            return transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ])
        else:  # val
            return transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # 数据移动到设备
            degraded_images = batch['degraded_image'].to(self.device)  # [B, 1, 3, H, W]
            clean_images = batch['clean_image'].to(self.device)  # [B, 1, 3, H, W]

            # 调整形状 [B, 1, 3, H, W] -> [B, 3, H, W]
            degraded_images = degraded_images.squeeze(1)
            clean_images = clean_images.squeeze(1)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(degraded_images)

            # 计算损失 (添加scale)
            loss = self.scaler * self.criterion(outputs, clean_images)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config['training'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )

            self.optimizer.step()

            # 更新EMA
            if self.ema:
                self.ema.update()

            # 更新统计信息
            epoch_loss += loss.item()
            self.global_step += 1

            # 打印进度
            if (batch_idx + 1) % self.config['training'].get('print_freq', 10) == 0:
                batch_time = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * self.config['data']['train_batch_size'] / batch_time
                current_lr = self.optimizer.param_groups[0]['lr']

                self.logger.info(
                    f"Epoch [{self.current_epoch}/{self.config['training']['epochs']}] "
                    f"Batch [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {loss.item():.6f} "
                    f"LR: {current_lr:.2e} "
                    f"Speed: {samples_per_sec:.1f} samples/sec"
                )

        epoch_loss /= num_batches
        epoch_time = time.time() - start_time

        return epoch_loss, epoch_time

    def validate(self):
        """验证"""
        self.logger.info(f"(>ω<) Start validation at epoch:{self.current_epoch}...")
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()

        val_loss = 0.0
        num_batches = len(self.val_loader)

        # 初始化metrics

        psnr = []
        ssim = []
        nmse = []

        with torch.no_grad():
            for batch in self.val_loader:
                degraded_images = batch['mixed'].to(self.device)
                clean_images = batch['clean_image'].to(self.device)

                degraded_images = degraded_images.squeeze(1)
                clean_images = clean_images.squeeze(1)

                #

                outputs = self.model(degraded_images)
                loss = self.criterion(outputs, clean_images)
                # calculate metrics
                psnr.append(calculate_psnr(outputs, clean_images))
                ssim.append(calculate_ssim(outputs, clean_images))
                nmse.append(calculate_nmse(outputs, clean_images))
                #
                val_loss += loss.item()

        avg_psnr = np.mean(np.array(psnr))
        avg_ssim = np.mean(np.array(ssim))
        avg_nmse = np.mean(np.array(nmse))

        self.logger.info(f"validation results: PSNR-{avg_psnr:.4f} SSIM-{avg_ssim:.4f} NMSE-{avg_nmse:.4f} ...")

        val_loss /= num_batches

        if self.ema:
            self.ema.restore()

        return val_loss, avg_psnr, avg_ssim, avg_nmse

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_psnr': self.best_psnr,
            'config': self.config
        }

        if self.ema:
            checkpoint['ema_shadow'] = self.ema.shadow

        checkpoint_path = self.checkpoint_dir / f'{self.model_name}_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.6f} PSNR:{self.best_psnr}")

        # # 定期保存
        # if self.current_epoch % self.config['training'].get('save_interval', 10) == 0:
        #     epoch_path = self.checkpoint_dir / f'{self.model_name}_epoch_{self.current_epoch}.pth'
        #     torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.ema and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_psnr = checkpoint['best_psnr']

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")

    def train(self):
        """训练主循环"""
        self.logger.info("(>ω<) Starting training...")

        # 构建所有组件
        self.build_model()
        self.build_optimizer()
        self.build_criterion()
        self.build_dataloaders()

        # 加载检查点（如果存在）
        if self.config['training'].get('resume'):
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_latest.pth'
            self.load_checkpoint(checkpoint_path)

        # 训练循环
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch

            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            self.logger.info(f"{'=' * 50}")

            # 训练
            train_loss, train_time = self.train_epoch()

            # 验证
            val_loss , avg_psnr, avg_ssim, avg_nmse = self.validate()

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 记录结果
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch + 1} Summary: "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"avg PSNR: {avg_psnr:.4f}, "
                f"avg SSIM: {avg_ssim:.4f}, "
                f"avg NMSE: {avg_nmse:.4f}, "
                f"LR: {current_lr:.2e}, "
                f"Time: {train_time:.2f}s"
            )

            # 保存检查点
            is_best = avg_psnr > self.best_psnr
            if is_best:
                self.best_val_loss = val_loss
                self.best_psnr = avg_psnr
                self.best_ssim = avg_ssim
                self.best_nmse = avg_nmse
                self.logger.info("(@ω@) got new best results!!!")
            self.save_checkpoint(is_best=is_best)

        self.logger.info("(>ω<) Training completed!")

def init_training(modelname):
    # 对比
    if modelname == 'UNet_Small':
        config = get_config_unet_small()
    elif modelname == 'UNet':
        config = get_config_unet()
    elif modelname == 'SOTA_DHA':
        config = get_config_DHA()
    elif modelname == "AdaIR":
        config = get_config_adair_base()
    elif modelname == "GUNet":
        config = get_config_gunet()
    elif modelname == "VIRNet":
        from configs.configs_plain_VIRNet import get_config_virnet
        config = get_config_virnet()
    # 本文方法
    elif modelname == 'PalmIR':
        config = get_config_palmir_base()
    elif modelname == 'PalmIR_Ablation':
        config = get_config_palmir_ablation()
    else:
        raise ValueError(f"Unknown model name: {modelname}")
    trainer = PalmprintTrainer(config)
    trainer.train()

if __name__ == "__main__":
    init_training('PalmIR')
