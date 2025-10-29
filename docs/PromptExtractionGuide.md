# ComfyUI PNG图像提示词提取指南

## 概述

本指南详细说明了如何从ComfyUI生成的PNG图像中提取工作流信息和提示词(prompt)。ComfyUI在保存图像时会自动将完整的工作流信息嵌入到PNG的元数据中，我们可以通过解析这些元数据来恢复原始的提示词和工作流配置。

## 工作原理

ComfyUI的PNG图像元数据中通常包含以下关键信息：

1. `workflow` - 完整的工作流定义，包含所有节点配置
2. `prompt` - 简化的提示词信息

我们的提取流程如下：

1. 读取PNG图像的元数据
2. 解析JSON格式的工作流数据
3. 提取并分析CLIPEncoder类节点中的提示词
4. 输出结构化的提示词信息

## 关键算法

### 1. 提取元数据

```python
with Image.open(image_path) as img:
    # 获取PNG元数据
    if hasattr(img, 'info'):
        # 优先检查workflow元数据
        if 'workflow' in img.info:
            prompt_data = img.info['workflow']
        # 其次检查prompt元数据
        elif 'prompt' in img.info:
            prompt_data = img.info['prompt']
```

### 2. 处理嵌套工作流结构

```python
def process_nested_workflow(prompt_json):
    """处理嵌套结构的workflow JSON，提取nodes字段中的实际节点"""
    # 检查是否有nodes字段
    if isinstance(prompt_json, dict) and 'nodes' in prompt_json and isinstance(prompt_json['nodes'], list):
        # 转换为以节点ID为键的字典
        node_dict = {}
        for node in prompt_json['nodes']:
            if isinstance(node, dict) and 'id' in node:
                node_id = str(node['id'])
                # 标准化节点数据，提取class_type和inputs
                if 'type' in node:
                    node['class_type'] = node['type']  # ComfyUI有时使用type而不是class_type
                # 处理inputs数组
                if 'inputs' in node and isinstance(node['inputs'], list):
                    # 转换为字典格式
                    # ...
                node_dict[node_id] = node
        return node_dict
    else:
        return prompt_json
```

### 3. 识别CLIPEncoder节点

```python
clip_encoder_keywords = ['CLIPTextEncode', 'CLIPEncoder', 'ClipTextEncode', 'CLIPTextEncodeSDXL', 
                       'CLIPTextEncodeSD3', 'CLIPVisionEncode', 'CLIP', 'CLIPLoader', 
                       'textencoder', 'TextEncoder', 'TEXT_ENCODER', 'CLIPTextEncodeFlux']

def is_clip_encoder_node(node_type):
    return any(keyword in node_type for keyword in clip_encoder_keywords)
```

### 4. 提取提示词

```python
def extract_prompts_from_node(node_data):
    """从单个节点中提取提示词"""
    prompts = []
    
    # 优先从widgets_values提取（ComfyUI的主要存储位置）
    if 'widgets_values' in node_data and isinstance(node_data['widgets_values'], list):
        for value in node_data['widgets_values']:
            if isinstance(value, str) and value.strip():
                prompts.append(value)
    
    # 其次从inputs提取
    if 'inputs' in node_data and isinstance(node_data['inputs'], dict):
        text_keywords = ['text', 'prompt', 'positive', 'negative']
        for key, value in node_data['inputs'].items():
            if any(kw in key.lower() for kw in text_keywords) and isinstance(value, str):
                prompts.append(value)
    
    return prompts
```

## 支持的节点类型

我们的提取工具支持以下CLIPEncoder相关节点类型：

1. **CLIPTextEncode** - 标准CLIP文本编码器
2. **CLIPTextEncodeFlux** - Flux模型的CLIP编码器
3. **CLIPTextEncodeSDXL** - SDXL模型的CLIP编码器
4. **TextEncodeQwenImageEditPlus** - QwenEdit相关的文本编码器
5. **xiaodu:QwenEditEncode** - 小度QwenEdit编码器
6. **CR Text** - ControlRoom的文本节点
7. **LayerUtility: TextJoin** - 文本拼接节点

## 提示词输入方式

在ComfyUI中，提示词的输入方式主要有：

1. **widgets_values字段** - 最常见的存储方式，节点界面上的输入框内容
2. **inputs字典** - 节点间的连接或参数设置
3. **嵌套节点引用** - 一个节点引用另一个节点的输出作为提示词

## 错误处理

在提取过程中，我们需要处理以下常见错误：

1. JSON解析错误 - 元数据格式可能不正确
2. 嵌套结构不规范 - 工作流数据可能有不同的组织方式
3. 编码问题 - 提示词可能包含特殊字符

## 集成到节点

要将此功能集成到自定义节点中，只需导入必要的库并添加提取方法：

```python
import json
from PIL import Image

def extract_prompts_from_image(image_path):
    """从PNG图像中提取所有提示词"""
    all_prompts = []
    try:
        with Image.open(image_path) as img:
            # 提取元数据和处理逻辑
            # ...
    except Exception as e:
        print(f"提取提示词时出错: {e}")
    return all_prompts
```

## 最佳实践

1. 始终检查元数据是否存在，优雅处理不存在的情况
2. 使用try-except块捕获可能的解析错误
3. 处理不同版本ComfyUI生成的工作流格式差异
4. 考虑添加缓存机制，避免重复处理相同的图像

## 示例

完整的提取过程示例可以参考`read_json_workflow.py`脚本，该脚本提供了完整的实现和详细的输出。

## 注意事项

1. 某些特殊插件节点可能有自定义的提示词存储方式
2. 工作流文件可能会随着ComfyUI版本更新而变化格式
3. 大型工作流可能包含大量节点，需要优化处理性能

---

*本指南由AFL插件团队维护，如有更新请参考最新文档。*