import torch
import numpy as np
from PIL import Image, ImageOps
import cv2

class MaskBoxCropNode:
    """
    根据mask裁剪图像并调整大小
    """
    
    CATEGORY = "AFL/emoji"
    DESCRIPTION = "根据mask裁剪图像区域并调整大小"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "resize_mode": (["lanczos", "nearest-exact", "bilinear", "bicubic"], {"default": "lanczos"}),
            },
            "optional": {
                "grow_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.05, "tooltip": "裁剪区域的扩展倍数，1.0表示不扩展，大于1.0表示按比例扩大"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROPBOX", "MASK")
    RETURN_NAMES = ("cropped_image", "crop_box", "cropped_mask")
    FUNCTION = "crop_and_resize"

    def _tensor_to_pil(self, tensor):
        """将ComfyUI的tensor转换为PIL图像"""
        # 确保输入tensor是正确的数据类型和形状
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
            
        # 处理不同的tensor形状
        if len(tensor.shape) == 4:
            # 标准的4D tensor (batch, height, width, channels)
            img_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        elif len(tensor.shape) == 3:
            # 3D tensor (height, width, channels)
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
            
        return Image.fromarray(img_np)
    
    def _tensor_to_pil_mask(self, mask):
        """将ComfyUI的mask tensor转换为PIL图像"""
        # 确保输入mask是正确的数据类型
        if mask.dtype != torch.float32:
            mask = mask.float()
            
        # 处理不同的mask形状
        if len(mask.shape) == 4:
            # 标准的4D mask tensor (batch, height, width, channels)
            mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        elif len(mask.shape) == 3:
            # 3D mask tensor (batch, height, width) 或 (height, width, channels)
            if mask.shape[0] == 1:  # (1, height, width)
                mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            else:  # (height, width, channels) 或其他情况
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        elif len(mask.shape) == 2:
            # 2D mask tensor (height, width)
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            
        # 确保mask_np是2D数组
        if len(mask_np.shape) > 2:
            # 如果是3D数组，取第一个通道
            mask_np = mask_np[:, :, 0] if mask_np.shape[2] > 1 else mask_np[:, :, 0]
            
        return Image.fromarray(mask_np, mode='L')

    def _pil_to_tensor(self, pil_image):
        """将PIL图像转换为ComfyUI的tensor"""
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)[None,]
    
    def _pil_to_mask(self, pil_mask):
        """将PIL mask转换为ComfyUI的mask tensor"""
        mask_np = np.array(pil_mask).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np)[None,]
    
    def crop_and_resize(self, image, mask, resize_mode, grow_factor=1.0):
        # 将输入转换为PIL图像
        pil_image = self._tensor_to_pil(image)
        pil_mask = self._tensor_to_pil_mask(mask)
        
        # 确保mask是二值图像
        pil_mask = pil_mask.convert('L')
        
        # 获取mask的边界框
        bbox = pil_mask.getbbox()
        if bbox is None:
            # 如果没有找到mask，返回整个图像
            bbox = (0, 0, pil_image.width, pil_image.height)
        
        # 紧密贴合mask边缘计算裁剪区域
        x1, y1, x2, y2 = bbox
        
        # 计算边界框的宽度和高度
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # 确保裁剪区域是正方形，使用边界框的最大维度
        max_dim = max(bbox_width, bbox_height)
        # 使用grow_factor参数来扩展裁剪区域
        target_size = int(max_dim * grow_factor)  # 按倍数扩展
            
        # 计算正方形的中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 计算正方形的边界
        half_size = target_size // 2
        square_x1 = center_x - half_size
        square_y1 = center_y - half_size
        square_x2 = center_x + half_size
        square_y2 = center_y + half_size
        
        # 处理边界超出图像的情况
        pad_left = max(0, -square_x1)
        pad_top = max(0, -square_y1)
        pad_right = max(0, square_x2 - pil_image.width)
        pad_bottom = max(0, square_y2 - pil_image.height)
        
        # 如果需要padding，则对图像和mask都进行padding
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            pil_image = ImageOps.expand(pil_image, (pad_left, pad_top, pad_right, pad_bottom), fill=(255, 255, 255))
            pil_mask = ImageOps.expand(pil_mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
            
            # 调整坐标
            square_x1 += pad_left
            square_y1 += pad_top
            square_x2 += pad_left
            square_y2 += pad_top
        
        # 裁剪正方形区域
        crop_box = (square_x1, square_y1, square_x2, square_y2)
        cropped_image = pil_image.crop(crop_box)
        cropped_mask = pil_mask.crop(crop_box)
        
        # Resize到1024*1024
        resample_filter = {
            "lanczos": Image.LANCZOS,
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC
        }.get(resize_mode, Image.LANCZOS)
        
        resized_image = cropped_image.resize((1024, 1024), resample_filter)
        resized_mask = cropped_mask.resize((1024, 1024), resample_filter)
        
        # 转换回tensor
        output_image = self._pil_to_tensor(resized_image)
        output_mask = self._pil_to_mask(resized_mask)
        
        # 返回crop_box信息以便还原使用
        crop_info = {
            "original_coords": crop_box,
            "padded_size": (pil_image.width, pil_image.height),
            "original_image_size": (image.shape[2], image.shape[1]),  # width, height
            "pad_info": (pad_left, pad_top, pad_right, pad_bottom)
        }
        
        return (output_image, crop_info, output_mask)


class ImageRestoreNode:
    """
    将处理后的图像粘贴回原来的图像中
    """
    
    CATEGORY = "AFL/emoji"
    DESCRIPTION = "将处理后的图像粘贴回原图"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "crop_box": ("CROPBOX",),
                "blur_amount": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore_image"
    
    def _tensor_to_pil(self, tensor):
        """将ComfyUI的tensor转换为PIL图像"""
        # 确保输入tensor是正确的数据类型和形状
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
            
        # 处理不同的tensor形状
        if len(tensor.shape) == 4:
            # 标准的4D tensor (batch, height, width, channels)
            img_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        elif len(tensor.shape) == 3:
            # 3D tensor (height, width, channels)
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
            
        return Image.fromarray(img_np)
    
    def _tensor_to_pil_mask(self, mask):
        """将ComfyUI的mask tensor转换为PIL图像"""
        # 确保输入mask是正确的数据类型
        if mask.dtype != torch.float32:
            mask = mask.float()
            
        # 处理不同的mask形状
        if len(mask.shape) == 4:
            # 标准的4D mask tensor (batch, height, width, channels)
            mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        elif len(mask.shape) == 3:
            # 3D mask tensor (batch, height, width) 或 (height, width, channels)
            if mask.shape[0] == 1:  # (1, height, width)
                mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            else:  # (height, width, channels) 或其他情况
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        elif len(mask.shape) == 2:
            # 2D mask tensor (height, width)
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            
        # 确保mask_np是2D数组
        if len(mask_np.shape) > 2:
            # 如果是3D数组，取第一个通道
            mask_np = mask_np[:, :, 0] if mask_np.shape[2] > 1 else mask_np[:, :, 0]
            
        return Image.fromarray(mask_np, mode='L')

    def _pil_to_tensor(self, pil_image):
        """将PIL图像转换为ComfyUI的tensor"""
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)[None,]
    
    def restore_image(self, original_image, processed_image, crop_box, blur_amount):
        # 将输入转换为PIL图像
        original_pil = self._tensor_to_pil(original_image)
        processed_pil = self._tensor_to_pil(processed_image)
        
        # 获取裁剪信息
        original_coords = crop_box["original_coords"]
        padded_size = crop_box["padded_size"]
        original_image_size = crop_box["original_image_size"]
        pad_info = crop_box["pad_info"]
        
        pad_left, pad_top, pad_right, pad_bottom = pad_info
        
        # 调整processed_image大小以匹配裁剪区域
        crop_width = original_coords[2] - original_coords[0]
        crop_height = original_coords[3] - original_coords[1]
        resized_processed = processed_pil.resize((crop_width, crop_height), Image.LANCZOS)
        
        # 创建一个与填充后图像相同大小的图像副本
        restored_image = original_pil.copy()
        if padded_size != (original_image_size[0], original_image_size[1]):
            # 如果之前进行了padding，我们需要创建一个填充后的图像
            restored_image = Image.new("RGB", padded_size, (255, 255, 255))
            # 粘贴原始图像的有效区域
            orig_region = (
                pad_left, 
                pad_top, 
                pad_left + original_image_size[0], 
                pad_top + original_image_size[1]
            )
            restored_image.paste(original_pil, orig_region)
        
        # 保存原始填充后图像用于边缘模糊
        padded_original = restored_image.copy()
        
        # 将处理后的图像粘贴回去
        restored_image.paste(resized_processed, original_coords[:2])
        
        # 应用边缘模糊效果
        if blur_amount > 0:
            restored_image = self._apply_edge_blur(
                restored_image, 
                padded_original, 
                original_coords, 
                blur_amount,
                pad_info,
                original_image_size
            )
        
        # 移除padding回到原始尺寸
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            restored_image = restored_image.crop((
                pad_left, 
                pad_top, 
                pad_left + original_image_size[0], 
                pad_top + original_image_size[1]
            ))
        
        # 转换回tensor
        output_image = self._pil_to_tensor(restored_image)
        return (output_image,)
    
    def _apply_edge_blur(self, restored_image, original_image, crop_coords, blur_amount, pad_info, original_size):
        """应用边缘模糊效果
        blur_amount: 边缘纯白色的像素数量（从裁剪框边缘向内延伸的宽度，完全透明区域，显示原始图像）
        """
        # 将PIL图像转换为numpy数组进行处理
        restored_np = np.array(restored_image)
        original_np = np.array(original_image)
        
        x1, y1, x2, y2 = crop_coords
        pad_left, pad_top, pad_right, pad_bottom = pad_info
        orig_width, orig_height = original_size
        
        # 计算在填充后图像中的实际坐标
        actual_x1 = x1
        actual_y1 = y1
        actual_x2 = x2
        actual_y2 = y2
        
        # 创建基础mask，初始化为全0（完全使用处理后图像）
        base_mask = np.zeros(restored_np.shape[:2], dtype=np.uint8)
        
        # 定义边缘纯白色区域（完全透明，显示原始图像）
        # 边缘区域：从裁剪框边缘向内延伸blur_amount像素的区域
        if blur_amount > 0:
            # 计算内部区域边界（从边缘向内延伸blur_amount像素后的边界）
            inner_x1 = actual_x1 + blur_amount
            inner_y1 = actual_y1 + blur_amount
            inner_x2 = actual_x2 - blur_amount
            inner_y2 = actual_y2 - blur_amount
            
            # 确保内部区域边界有效（不超出裁剪框范围）
            inner_x1 = max(actual_x1, min(inner_x1, actual_x2))
            inner_y1 = max(actual_y1, min(inner_y1, actual_y2))
            inner_x2 = max(actual_x1, min(inner_x2, actual_x2))
            inner_y2 = max(actual_y1, min(inner_y2, actual_y2))
            
            # 定义过渡区域边界（从边缘向内延伸2*blur_amount像素，用于模糊过渡）
            transition_x1 = actual_x1 + blur_amount
            transition_y1 = actual_y1 + blur_amount
            transition_x2 = actual_x2 - blur_amount
            transition_y2 = actual_y2 - blur_amount
            
            # 创建过渡区域的mask（用于后续模糊）
            transition_mask = np.zeros(restored_np.shape[:2], dtype=np.uint8)
            
            # 定义纯白色边缘区域（完全显示原始图像）
            # 1. 左边边缘：从左边界到inner_x1
            if actual_x1 < inner_x1:
                base_mask[:, actual_x1:inner_x1] = 255
            
            # 2. 右边边缘：从inner_x2到右边界
            if inner_x2 < actual_x2:
                base_mask[:, inner_x2:actual_x2] = 255
            
            # 3. 顶部边缘：从顶部边界到inner_y1
            if actual_y1 < inner_y1:
                base_mask[actual_y1:inner_y1, actual_x1:actual_x2] = 255
            
            # 4. 底部边缘：从inner_y2到actual_y2
            if inner_y2 < actual_y2:
                base_mask[inner_y2:actual_y2, actual_x1:actual_x2] = 255
            
            # 定义过渡区域（用于创建平滑过渡效果）
            # 这个区域将在后面应用高斯模糊
            # 1. 左侧过渡区：从inner_x1到transition_x1 + blur_amount
            left_transition_end = min(transition_x1 + blur_amount, actual_x2)
            if inner_x1 < left_transition_end:
                transition_mask[:, inner_x1:left_transition_end] = 255
            
            # 2. 右侧过渡区：从transition_x2 - blur_amount到inner_x2
            right_transition_start = max(transition_x2 - blur_amount, actual_x1)
            if right_transition_start < inner_x2:
                transition_mask[:, right_transition_start:inner_x2] = 255
            
            # 3. 顶部过渡区：从inner_y1到transition_y1 + blur_amount
            top_transition_end = min(transition_y1 + blur_amount, actual_y2)
            if inner_y1 < top_transition_end:
                transition_mask[inner_y1:top_transition_end, actual_x1:actual_x2] = 255
            
            # 4. 底部过渡区：从transition_y2 - blur_amount到inner_y2
            bottom_transition_start = max(transition_y2 - blur_amount, actual_y1)
            if bottom_transition_start < inner_y2:
                transition_mask[bottom_transition_start:inner_y2, actual_x1:actual_x2] = 255
            
            # 清除内部区域（中心部分）为0，确保内部区域完全使用处理后图像
            if inner_x1 < inner_x2 and inner_y1 < inner_y2:
                base_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0
                transition_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0
        
        # 应用高斯模糊来创建从纯白到纯黑的平滑过渡
        # 只对过渡区域应用模糊，保留纯白色边缘区域
        if blur_amount > 0:
            # 模糊半径为blur_amount，确保过渡平滑
            # 使用奇数大小的内核以获得更好的效果
            kernel_size = 2 * blur_amount + 1
            
            # 只对过渡区域应用模糊
            blurred_transition = cv2.GaussianBlur(transition_mask, (kernel_size, kernel_size), 0)
            
            # 创建最终mask：纯白色边缘区域保持不变，过渡区域使用模糊后的结果
            # 先复制基础mask（包含纯白色边缘区域）
            final_mask = base_mask.copy()
            
            # 在过渡区域应用模糊效果，但确保不覆盖纯白色边缘区域
            # 只在base_mask为0且transition_mask不为0的区域应用模糊
            transition_area = (base_mask == 0) & (transition_mask > 0)
            final_mask[transition_area] = blurred_transition[transition_area]
        else:
            final_mask = base_mask
        
        # 归一化到0-1范围
        final_mask = final_mask.astype(np.float32) / 255.0
        
        # 扩展mask维度以匹配图像
        final_mask = np.stack([final_mask] * 3, axis=-1)
        
        # 应用混合：
        # - mask值为0：完全使用处理后的图像 (restored_np)
        # - mask值为1：完全使用原始图像 (original_np)
        # - 中间值：混合使用
        restored_np = (restored_np * (1 - final_mask) + original_np * final_mask).astype(np.uint8)
        
        # 转换回PIL图像
        return Image.fromarray(restored_np)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AFL:MaskBoxCropNode": MaskBoxCropNode,
    "AFL:ImageRestoreNode": ImageRestoreNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:MaskBoxCropNode": "Target box crop",
    "AFL:ImageRestoreNode": "Target restore"
}