# -*- coding: utf-8 -*-
# Green_graffiti.py
# 功能：严格按照两步处理流程：1) 将图像转换为绿色线条 2) 仅在mask范围内覆盖到原图上

from __future__ import annotations
import numpy as np
import torch
from PIL import Image

try:
    import cv2
except Exception as e:
    raise RuntimeError("需要安装 opencv-python: pip install opencv-python") from e


class GreenGraffiti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),    # 用于生成绿色线条的图像
                "image2": ("IMAGE",),    # 要被覆盖的图像
                "opacity": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),  # 不透明度 (0-100)
            },
            "optional": {
                "mask": ("MASK",),      # 遮罩图像（可选）
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "AFL/Mask"
    
    _DEFAULT_WHITE_TH = 240
    
    def _step1_generate_green_lineart(self, image_np):
        """
        Step 1: 精确实现LineartColorifyV2.py中thickness=0且color=green的处理逻辑
        image_np: 输入图像 numpy数组 (H,W,3), 范围[0,255]
        return: 绿色线条图像 numpy数组 (H,W,3), 范围[0,255]
        """
        # 将numpy数组转换为PIL图像（严格按照LineartColorifyV2的处理流程）
        pil_img = Image.fromarray(image_np)
        
        # 计算灰度图
        gray = np.array(pil_img.convert("L"))
        
        # 计算透明度通道（thickness=0时的soft_alpha逻辑）
        soft_alpha = 255 - gray
        soft_alpha = np.where(gray >= self._DEFAULT_WHITE_TH, 0, soft_alpha).astype(np.uint8)
        
        # 获取图像尺寸
        h, w = gray.shape
        
        # 创建空白图像用于绘制绿色线条
        green_lineart = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 设置绿色线条（BGR格式）
        green_lineart[:, :, 1] = 255  # 绿色通道设置为255
        
        # 将alpha通道应用到绿色线条上
        # 只保留有alpha值的区域的绿色
        green_lineart = green_lineart * (soft_alpha[:, :, np.newaxis] / 255.0)
        green_lineart = green_lineart.astype(np.uint8)
        
        return green_lineart, soft_alpha
    
    def _prepare_mask(self, mask_tensor, image_shape):
        """
        准备mask，确保格式正确且与图像尺寸匹配
        """
        # 将tensor转换为numpy数组
        mask_np = mask_tensor.detach().cpu().numpy()
        
        # 处理不同维度的mask
        if mask_np.ndim == 3:
            # 处理3D mask：[H,W,C] 或 [B,H,W]
            if mask_np.shape[-1] == 1 or mask_np.shape[0] == 1:
                # 移除单通道或单批次维度
                mask_np = mask_np.squeeze()
            else:
                # 对多通道mask取平均值
                mask_np = mask_np.mean(axis=-1)
        elif mask_np.ndim == 1:
            # 处理异常情况，创建空mask
            mask_np = np.zeros(image_shape[:2])
        
        # 确保mask是二维的
        if mask_np.ndim != 2:
            mask_np = np.zeros(image_shape[:2])
        
        # 归一化mask到0-1范围
        if mask_np.max() > 1.0:
            mask_np = mask_np / 255.0
        
        # 确保mask与图像尺寸匹配
        if mask_np.shape != image_shape[:2]:
            mask_np = cv2.resize(mask_np, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 确保mask在0-1范围内
        mask_np = np.clip(mask_np, 0, 1)
        
        return mask_np

    def process(self, image1: torch.Tensor, image2: torch.Tensor, opacity: int, mask: torch.Tensor = None):
        """
        主处理函数：严格按照两步处理流程
        Step 1: 使用image1生成绿色线条图像
        Step 2: 在mask范围内将绿色线条覆盖到image2上，应用opacity参数
                如果mask为None，则直接覆盖整个图像
        """
        # 验证输入
        assert isinstance(image1, torch.Tensor) and image1.ndim == 4, "image1 必须为 [B,H,W,C] 的 Tensor"
        assert isinstance(image2, torch.Tensor) and image2.ndim == 4, "image2 必须为 [B,H,W,C] 的 Tensor"
        
        # 将不透明度转换为0-1范围
        opacity_normalized = opacity / 100.0
        
        batch_output = []
        
        # 处理每个批次的图像
        for i in range(image1.shape[0]):
            # 获取当前批次的图像1、图像2和对应的mask
            current_image1 = image1[i]
            # 确保image2有足够的批次，如果批次不足则重复使用最后一个
            current_image2 = image2[i] if i < image2.shape[0] else image2[-1]
            # 只有当mask不为None时才尝试获取current_mask
            current_mask = mask[i] if mask is not None and mask.ndim == 4 else mask
            
            # 转换tensor为numpy数组 (H,W,3), 范围[0,255]
            img1_np = (current_image1.detach().cpu().numpy() * 255).astype(np.uint8)
            img2_np = (current_image2.detach().cpu().numpy() * 255).astype(np.uint8)
            
            # =================================================================
            # Step 1: 使用image1生成绿色线条图像
            # =================================================================
            green_lineart, alpha_channel = self._step1_generate_green_lineart(img1_np)
            
            # =================================================================
            # Step 2: 准备mask并将绿色线条覆盖到image2的mask区域
            # =================================================================
            # 准备mask，如果mask为None则创建全1的mask（覆盖整个图像）
            if mask is None:
                # 创建全1的mask，表示覆盖整个图像
                mask_normalized = np.ones(img1_np.shape[:2])
            else:
                # 使用提供的mask
                mask_normalized = self._prepare_mask(current_mask, img1_np.shape)
            
            # 创建最终的alpha蒙版，结合mask区域和opacity设置
            # 这确保只在mask区域内应用绿色线条，并且可以通过opacity调整透明度
            final_alpha = (alpha_channel / 255.0) * opacity_normalized * mask_normalized
            
            # 将final_alpha扩展为3通道，用于图像混合
            final_alpha_3channel = np.stack([final_alpha, final_alpha, final_alpha], axis=2)
            
            # 混合图像：
            # - 在mask区域内：image2*(1-opacity) + 绿色线条*opacity
            # - 在mask区域外：保持image2不变
            img2_float = img2_np.astype(np.float32)
            green_float = green_lineart.astype(np.float32)
            
            # 应用混合
            blended_np = img2_float * (1 - final_alpha_3channel) + green_float * final_alpha_3channel
            blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
            
            # 转换回tensor格式
            blended_tensor = torch.from_numpy(blended_np.astype(np.float32) / 255.0)
            batch_output.append(blended_tensor)
        
        # 堆叠批次并返回
        out_tensor = torch.stack(batch_output, dim=0).to(image1.device)
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "AFL:GreenGraffiti": GreenGraffiti
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:GreenGraffiti": "GreenGraffiti"
}