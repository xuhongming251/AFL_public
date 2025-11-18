# -*- coding: utf-8 -*-
# Green_mask.py
# 功能：直接生成纯绿色图像，然后在mask范围内覆盖到原图上，支持不透明度调整

from __future__ import annotations
import numpy as np
import torch

try:
    import cv2
except Exception as e:
    raise RuntimeError("需要安装 opencv-python: pip install opencv-python") from e


class GreenMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),    # 输入图像
                "mask": ("MASK",),      # 遮罩图像
                "opacity": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),  # 不透明度 (0-100)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "AFL/Mask"
    
    def _step1_generate_pure_green(self, image_shape):
        """
        Step 1: 直接生成同尺寸的纯绿色图像
        image_shape: 输入图像形状 (H,W,3)
        return: 纯绿色图像 numpy数组 (H,W,3), 范围[0,255]
        """
        h, w = image_shape[:2]
        
        # 创建空白图像并设置为纯绿色
        pure_green = np.zeros((h, w, 3), dtype=np.uint8)
        pure_green[:, :, 1] = 255  # 将绿色通道设置为最大值(255)
        
        return pure_green
    
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

    def process(self, image: torch.Tensor, mask: torch.Tensor, opacity: int):
        """
        主处理函数
        Step 1: 生成纯绿色图像
        Step 2: 在mask范围内将纯绿色图像覆盖到原图上，应用opacity参数
        """
        # 验证输入
        assert isinstance(image, torch.Tensor) and image.ndim == 4, "image 必须为 [B,H,W,C] 的 Tensor"
        assert isinstance(mask, torch.Tensor), "mask 必须为 Tensor"
        
        # 将不透明度转换为0-1范围
        opacity_normalized = opacity / 100.0
        
        batch_output = []
        
        # 处理每个批次的图像
        for i in range(image.shape[0]):
            # 获取当前批次的图像和对应的mask
            current_image = image[i]
            current_mask = mask[i] if mask.ndim == 4 else mask
            
            # 转换tensor为numpy数组 (H,W,3), 范围[0,255]
            img_np = (current_image.detach().cpu().numpy() * 255).astype(np.uint8)
            
            # =================================================================
            # Step 1: 直接生成同尺寸的纯绿色图像
            # =================================================================
            pure_green = self._step1_generate_pure_green(img_np.shape)
            
            # =================================================================
            # Step 2: 准备mask并将纯绿色图像覆盖到原图的mask区域
            # =================================================================
            # 准备mask
            mask_normalized = self._prepare_mask(current_mask, img_np.shape)
            
            # 创建最终的alpha蒙版，结合mask区域和opacity设置
            final_alpha = opacity_normalized * mask_normalized
            
            # 将final_alpha扩展为3通道，用于图像混合
            final_alpha_3channel = np.stack([final_alpha, final_alpha, final_alpha], axis=2)
            
            # 混合图像：
            # - 在mask区域内：原图*(1-opacity) + 纯绿色*opacity
            # - 在mask区域外：保持原图不变
            img_float = img_np.astype(np.float32)
            green_float = pure_green.astype(np.float32)
            
            # 应用混合
            blended_np = img_float * (1 - final_alpha_3channel) + green_float * final_alpha_3channel
            blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
            
            # 转换回tensor格式
            blended_tensor = torch.from_numpy(blended_np.astype(np.float32) / 255.0)
            batch_output.append(blended_tensor)
        
        # 堆叠批次并返回
        out_tensor = torch.stack(batch_output, dim=0).to(image.device)
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "AFL:GreenMask": GreenMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:GreenMask": "GreenMask"
}