import os
import torch
import hashlib
import json
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import folder_paths
import node_helpers
import comfy.utils

class AFL_LoadImage:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        try:
            files_list = os.listdir(input_dir)
            files = [f for f in files_list if os.path.isfile(os.path.join(input_dir, f))]
            files = folder_paths.filter_files_content_types(files, ["image"])
        except:
            pass
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            }
        }

    CATEGORY = "AFL/实用工具"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "prompts", "loras", "AFL_fast_text")
    FUNCTION = "load_image"
    
    def load_image(self, image):
        # 使用默认路径加载图像
        image_path = folder_paths.get_annotated_filepath(image)

        # 首先加载图像
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # 提取提示词
        prompts = self.extract_prompts_from_image(image_path)
        # 提取Lora信息
        loras = self.extract_loras_from_image(image_path)
        # 从图像JSON中提取AFL_fast_text节点内容
        fast_text_content = self.extract_fast_text_content(image_path)

        return (output_image, output_mask, prompts, loras, fast_text_content)
        
    def extract_prompts_from_image(self, image_path):
        """
        从PNG图像中提取所有提示词
        返回格式：包含所有提示词的字符串，每个提示词独立一行，空行分隔
        """
        all_prompts = []
        
        try:
            with Image.open(image_path) as img:
                # 检查是否有workflow或prompt元数据
                prompt_data = None
                
                # 尝试从PNG元数据中提取工作流信息
                if hasattr(img, 'info'):
                    if 'workflow' in img.info:
                        prompt_data = img.info['workflow']
                    elif 'prompt' in img.info:
                        prompt_data = img.info['prompt']
                
                # 如果找到数据，解析JSON
                if prompt_data:
                    # 确保是字符串格式
                    if isinstance(prompt_data, str):
                        try:
                            # 处理可能的压缩或编码问题
                            if prompt_data.startswith('{') and prompt_data.endswith('}'):
                                prompt_json = json.loads(prompt_data)
                            else:
                                # 尝试解码base64或其他格式（如果需要）
                                prompt_json = json.loads(prompt_data)
                        except json.JSONDecodeError:
                            # 如果解析失败，尝试将其作为原始字符串处理
                            all_prompts.append(f"原始工作流数据: {prompt_data[:100]}...")
                            return "\n\n".join(all_prompts)
                    else:
                        prompt_json = prompt_data
                    
                    # 处理嵌套工作流结构
                    prompt_json = self.process_nested_workflow(prompt_json)
                    
                    # 提取所有提示词
                    all_prompts = self.extract_all_prompts(prompt_json)
        
        except Exception as e:
            # 出错时返回错误信息，但不影响图像加载
            all_prompts.append(f"提示词提取错误: {str(e)}")
        
        # 如果没有找到提示词，返回空字符串
        if not all_prompts:
            return ""
            
        # 返回用空行分隔的所有提示词
        return "\n\n".join(all_prompts)
    
    def process_nested_workflow(self, prompt_json):
        """处理嵌套结构的workflow JSON，提取nodes字段中的实际节点"""
        # 检查是否有nodes字段
        if isinstance(prompt_json, dict) and 'nodes' in prompt_json and isinstance(prompt_json['nodes'], list):
            # 转换为以节点ID为键的字典
            node_dict = {}
            for node in prompt_json['nodes']:
                if isinstance(node, dict) and 'id' in node:
                    node_id = str(node['id'])
                    # 标准化节点数据
                    if 'type' in node and 'class_type' not in node:
                        node['class_type'] = node['type']
                    node_dict[node_id] = node
            return node_dict
        else:
            return prompt_json
    
    def extract_all_prompts(self, prompt_json):
        """从处理后的工作流数据中提取所有提示词"""
        all_prompts = []
        clip_encoder_keywords = ['CLIPTextEncode', 'CLIPEncoder', 'ClipTextEncode', 'CLIPTextEncodeSDXL', 
                               'CLIPTextEncodeSD3', 'CLIPVisionEncode', 'CLIP', 'CLIPLoader', 
                               'textencoder', 'TextEncoder', 'TEXT_ENCODER', 'CLIPTextEncodeFlux']
        
        # 遍历所有节点
        if isinstance(prompt_json, dict):
            for node_id, node_data in prompt_json.items():
                if not isinstance(node_data, dict):
                    continue
                
                # 获取节点类型
                node_type = node_data.get('class_type', node_data.get('type', ''))
                
                # 检查是否为CLIPEncoder类型节点
                is_clip_node = any(keyword in node_type for keyword in clip_encoder_keywords)
                
                # 从widgets_values中提取提示词（ComfyUI的主要存储位置）
                if 'widgets_values' in node_data and isinstance(node_data['widgets_values'], list):
                    for i, value in enumerate(node_data['widgets_values']):
                        if isinstance(value, str) and value.strip():
                            # 过滤掉路径和文件名类的字符串
                            if len(value) > 0 and (len(value) <= 5 or ('\\' not in value and '/' not in value)):
                                if is_clip_node:
                                    all_prompts.append(f"[{node_type}] {value}")
                                else:
                                    # 对于非clip节点，也提取有价值的文本
                                    text_keywords = ['prompt', 'text', '提示', '描述']
                                    if any(kw in str(node_type).lower() for kw in text_keywords) or \
                                       any(kw in value.lower() for kw in text_keywords):
                                        all_prompts.append(f"[{node_type}] {value}")
                
                # 从inputs中提取提示词
                if 'inputs' in node_data and isinstance(node_data['inputs'], dict):
                    for key, value in node_data['inputs'].items():
                        if isinstance(value, str) and value.strip():
                            text_keywords = ['text', 'prompt', 'positive', 'negative']
                            if any(kw in key.lower() for kw in text_keywords):
                                all_prompts.append(f"[{node_type}] {value}")
        
        return all_prompts
        
    def extract_loras_from_image(self, image_path):
        """
        从PNG图像中提取所有Lora信息
        返回格式：每个Lora信息独立一行，空行分隔
        """
        all_loras = []
        
        try:
            with Image.open(image_path) as img:
                # 检查是否有workflow或prompt元数据
                prompt_data = None
                
                # 尝试从PNG元数据中提取工作流信息
                if hasattr(img, 'info'):
                    if 'workflow' in img.info:
                        prompt_data = img.info['workflow']
                    elif 'prompt' in img.info:
                        prompt_data = img.info['prompt']
                
                # 如果找到数据，解析JSON
                if prompt_data:
                    # 确保是字符串格式
                    if isinstance(prompt_data, str):
                        try:
                            # 处理可能的压缩或编码问题
                            if prompt_data.startswith('{') and prompt_data.endswith('}'):
                                prompt_json = json.loads(prompt_data)
                            else:
                                # 尝试解码base64或其他格式（如果需要）
                                prompt_json = json.loads(prompt_data)
                        except json.JSONDecodeError:
                            # 如果解析失败，尝试将其作为原始字符串处理
                            return "Lora提取错误: 无法解析JSON数据"
                    else:
                        prompt_json = prompt_data
                    
                    # 处理嵌套工作流结构
                    prompt_json = self.process_nested_workflow(prompt_json)
                    
                    # 提取所有Lora信息
                    all_loras = self.extract_all_loras(prompt_json)
            
        except Exception as e:
            # 出错时返回错误信息，但不影响图像加载
            return f"Lora提取错误: {str(e)}"
        
        # 如果没有找到Lora，返回空字符串
        if not all_loras:
            return ""
            
        # 格式化输出，每个Lora信息独立一行，空行分隔
        formatted_loras = []
        for i, lora_path in enumerate(all_loras, 1):
            formatted_loras.append(f"Lora{i}:{lora_path}")
            
        return "\n\n".join(formatted_loras)
    
    def extract_all_loras(self, prompt_json):
        """
        从处理后的工作流数据中提取所有Lora模型路径
        """
        all_loras = []
        
        # Lora相关节点的关键词
        lora_keywords = ['LoraLoader', 'Lora', 'Load LoRA', 'Load Lora', 'lora', 'LoRA']
        
        # 遍历所有节点
        if isinstance(prompt_json, dict):
            for node_id, node_data in prompt_json.items():
                if not isinstance(node_data, dict):
                    continue
                
                # 获取节点类型
                node_type = node_data.get('class_type', node_data.get('type', ''))
                
                # 检查是否为Lora相关节点
                is_lora_node = any(keyword in node_type for keyword in lora_keywords)
                
                if is_lora_node:
                    # 尝试从widgets_values中提取Lora路径
                    if 'widgets_values' in node_data and isinstance(node_data['widgets_values'], list):
                        # Lora路径通常是第一个参数
                        if len(node_data['widgets_values']) > 0:
                            lora_value = node_data['widgets_values'][0]
                            if isinstance(lora_value, str) and ('.safetensors' in lora_value or '.ckpt' in lora_value):
                                # 规范化路径分隔符为\
                                lora_path = lora_value.replace('/', '\\')
                                all_loras.append(lora_path)
                    
                    # 尝试从inputs中提取Lora路径
                    if 'inputs' in node_data and isinstance(node_data['inputs'], dict):
                        for key, value in node_data['inputs'].items():
                            # Lora路径通常在model_name, lora_name, name等字段
                            if any(kw in key.lower() for kw in ['model', 'name', 'lora']) and isinstance(value, str):
                                if '.safetensors' in value or '.ckpt' in value:
                                    # 规范化路径分隔符为\
                                    lora_path = value.replace('/', '\\')
                                    all_loras.append(lora_path)
        
        # 去重
        return list(dict.fromkeys(all_loras))
    
    def extract_fast_text_content(self, image_path):
        """
        从PNG图像中提取所有AFL_fast_text节点的内容
        返回格式：包含所有AFL_fast_text节点文本内容的字符串
        """
        fast_text_contents = []
        
        try:
            with Image.open(image_path) as img:
                # 检查是否有workflow或prompt元数据
                prompt_data = None
                
                # 尝试从PNG元数据中提取工作流信息
                if hasattr(img, 'info'):
                    if 'workflow' in img.info:
                        prompt_data = img.info['workflow']
                    elif 'prompt' in img.info:
                        prompt_data = img.info['prompt']
                
                # 如果找到数据，解析JSON
                if prompt_data:
                    # 确保是字符串格式
                    if isinstance(prompt_data, str):
                        try:
                            # 处理可能的压缩或编码问题
                            if prompt_data.startswith('{') and prompt_data.endswith('}'):
                                prompt_json = json.loads(prompt_data)
                            else:
                                # 尝试解码base64或其他格式（如果需要）
                                prompt_json = json.loads(prompt_data)
                        except json.JSONDecodeError:
                            # 如果解析失败，尝试将其作为原始字符串处理
                            return "AFL_fast_text提取错误: 无法解析JSON数据"
                    else:
                        prompt_json = prompt_data
                    
                    # 处理嵌套工作流结构
                    prompt_json = self.process_nested_workflow(prompt_json)
                    
                    # 提取所有AFL_fast_text节点内容
                    fast_text_contents = self.extract_all_fast_text(prompt_json)
            
        except Exception as e:
            # 出错时返回错误信息，但不影响图像加载
            return f"AFL_fast_text提取错误: {str(e)}"
        
        # 如果没有找到AFL_fast_text内容，返回空字符串
        if not fast_text_contents:
            return ""
            
        # 返回用空行分隔的所有AFL_fast_text内容
        return "\n\n".join(fast_text_contents)
    
    def extract_all_fast_text(self, prompt_json):
        """从处理后的工作流数据中提取所有AFL_fast_text节点的内容，包括串联节点"""
        fast_text_contents = []
        
        # AFL_fast_text相关节点的关键词
        fast_text_keywords = ['AFL_fast_text', 'AFL_Fast_Text', 'AFL:Fast_Text', 'AFL_textbox']
        
        # 存储所有节点和它们的ID映射
        all_nodes = {}
        fast_text_nodes = {}
        
        # 第一次遍历：收集所有节点信息
        if isinstance(prompt_json, dict):
            for node_id, node_data in prompt_json.items():
                if isinstance(node_data, dict):
                    all_nodes[node_id] = node_data
                    # 检查是否为AFL_fast_text相关节点
                    node_type = node_data.get('class_type', node_data.get('type', ''))
                    if any(keyword in node_type for keyword in fast_text_keywords):
                        fast_text_nodes[node_id] = node_data
        
        # 第二次遍历：提取文本内容，支持串联节点
        for node_id, node_data in fast_text_nodes.items():
            # 先提取直接的文本内容
            direct_content = self._extract_node_direct_content(node_data)
            if direct_content:
                fast_text_contents.extend(direct_content)
            
            # 处理串联节点：解析inputs中对其他节点的引用
            if 'inputs' in node_data and isinstance(node_data['inputs'], dict):
                for key, value in node_data['inputs'].items():
                    # ComfyUI中节点引用通常是[node_id, output_index]
                    if isinstance(value, list) and len(value) >= 2 and isinstance(value[0], (str, int)):
                        source_node_id = str(value[0])
                        # 检查引用的节点是否存在且是fast_text节点
                        if source_node_id in fast_text_nodes:
                            # 递归获取引用节点的内容
                            source_content = self._extract_node_direct_content(fast_text_nodes[source_node_id])
                            if source_content:
                                fast_text_contents.extend(source_content)
        
        # 返回提取的文本内容列表（去重但保持顺序）
        return list(dict.fromkeys(fast_text_contents))
    
    def _extract_node_direct_content(self, node_data):
        """提取单个节点的直接文本内容"""
        contents = []
        
        # 从widgets_values中提取文本内容
        if 'widgets_values' in node_data and isinstance(node_data['widgets_values'], list):
            for value in node_data['widgets_values']:
                if isinstance(value, str) and value.strip():
                    contents.append(value)
        
        # 从inputs中提取文本内容
        if 'inputs' in node_data and isinstance(node_data['inputs'], dict):
            for key, value in node_data['inputs'].items():
                # 检查是否为字符串类型的输入（非节点引用）
                if isinstance(value, str) and value.strip():
                    if any(kw in key.lower() for kw in ['text', 'content', 'value']):
                        contents.append(value)
        
        # 从outputs中提取文本内容
        if 'outputs' in node_data and isinstance(node_data['outputs'], list):
            for output in node_data['outputs']:
                if isinstance(output, dict):
                    # 检查output中是否有widgets_values
                    if 'widgets_values' in output and isinstance(output['widgets_values'], list):
                        for value in output['widgets_values']:
                            if isinstance(value, str) and value.strip():
                                contents.append(value)
                    # 检查output中是否有直接的文本字段
                    elif any(kw in output for kw in ['text', 'content', 'value']):
                        for kw in ['text', 'content', 'value']:
                            if kw in output and isinstance(output[kw], str) and output[kw].strip():
                                contents.append(output[kw])
        
        return contents

    @classmethod
    def IS_CHANGED(s, image):
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            m = hashlib.sha256()
            with open(image_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        except:
            return ""

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"无效的图像文件: {image}"
        return True
    


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFL:LoadImage": AFL_LoadImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:LoadImage": "AFL_LoadImage"
}

