"""
火灾检测训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from ..models import FireConvLSTM, heatmap_to_prob


class Trainer:
    """
    FireConvLSTM 训练器

    Args:
        model: FireConvLSTM 模型
        device: 训练设备
        learning_rate: 学习率
        weight_decay: 权重衰减
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # 损失函数
        self.criterion = nn.BCELoss()

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_acc = 0
        self.best_epoch = 0

    def train_one_epoch(self, dataloader) -> Tuple[float, float]:
        """
        训练一个 epoch

        Args:
            dataloader: 训练数据加载器

        Returns:
            (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training")
        for frames, labels in pbar:
            frames = frames.to(self.device)  # (B, T, 3, 640, 640)
            labels = labels.float().to(self.device)  # (B,)

            self.optimizer.zero_grad()

            # Forward
            heatmap = self.model(frames)  # (B, 1, 20, 20)
            probs = heatmap_to_prob(heatmap)  # (B,)

            # Loss
            loss = self.criterion(probs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * frames.size(0)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += frames.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, dataloader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        验证

        Args:
            dataloader: 验证数据加载器

        Returns:
            (avg_loss, accuracy, all_probs, all_labels)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_probs = []
        all_labels = []

        for frames, labels in tqdm(dataloader, desc="Validating"):
            frames = frames.to(self.device)
            labels = labels.float().to(self.device)

            heatmap = self.model(frames)
            probs = heatmap_to_prob(heatmap)

            loss = self.criterion(probs, labels)

            total_loss += loss.item() * frames.size(0)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += frames.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return total_loss / total, correct / total, np.array(all_probs), np.array(all_labels)

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        save_dir: str = './checkpoints',
        save_best_only: bool = True
    ) -> Dict:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            save_best_only: 是否只保存最佳模型

        Returns:
            训练历史
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("开始训练")
        print("=" * 60)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # 训练
            train_loss, train_acc = self.train_one_epoch(train_loader)

            # 验证
            val_loss, val_acc, _, _ = self.validate(val_loader)

            # 学习率调整
            self.scheduler.step(val_loss)

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 保存模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1

            if is_best or not save_best_only:
                self.save_checkpoint(
                    save_dir / f'checkpoint_epoch_{epoch + 1}.pth',
                    epoch + 1,
                    val_acc
                )
                if is_best:
                    self.save_checkpoint(save_dir / 'best_model.pth', epoch + 1, val_acc)
                    print(f"  ★ 新最佳模型! Val Acc: {val_acc:.4f}")

        print("\n" + "=" * 60)
        print(f"训练完成! 最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print("=" * 60)

        return self.history

    def save_checkpoint(self, path: str, epoch: int, val_acc: float):
        """
        保存检查点

        Args:
            path: 保存路径
            epoch: 当前 epoch
            val_acc: 验证准确率
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        加载检查点

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.history = checkpoint.get('history', self.history)

        return checkpoint.get('epoch', 0)
