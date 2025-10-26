import torch
import node_helpers
import comfy.utils
import math

class AFL_Qweneditplus_FastuseV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                # 添加缩放控制选项
                "images_vl1": (["ini(384x384)", "NaN"], {"default": "ini(384x384)"}),
                "ref_latents1": (["ini(1024*1024)", "NaN"], {"default": "ini(1024*1024)"}),
                "images_vl2": (["ini(384x384)", "NaN"], {"default": "ini(384x384)"}),
                "ref_latents2": (["ini(1024*1024)", "NaN"], {"default": "ini(1024*1024)"}),
                "images_vl3": (["ini(384x384)", "NaN"], {"default": "ini(384x384)"}),
                "ref_latents3": (["ini(1024*1024)", "NaN"], {"default": "ini(1024*1024)"}),
                # 添加选择latent输出的选项
                "choose_latent": (["image1", "image2", "image3"], {"default": "image1"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "con_zero_out", "latent")
    FUNCTION = "execute"
    CATEGORY = "AFL/Qwen Edit"

    def execute(self, clip, prompt, vae=None, image1=None, image2=None, image3=None,
                images_vl1="ini(384x384)", ref_latents1="ini(1024*1024)",
                images_vl2="ini(384x384)", ref_latents2="ini(1024*1024)",
                images_vl3="ini(384x384)", ref_latents3="ini(1024*1024)",
                choose_latent="image1"):
        ref_latents = []
        # 不再保存所有图像的latent，只为用户选择的图像生成
        chosen_latent = None  # 只保存用户选择的latent
        # 现在没有none选项了，直接设置索引
        chosen_latent_index = 0 if choose_latent == "image1" else 1 if choose_latent == "image2" else 2
        
        images = [image1, image2, image3]
        images_vl = []
        # 获取每张图像的缩放配置
        images_vl_settings = [images_vl1, images_vl2, images_vl3]
        ref_latents_settings = [ref_latents1, ref_latents2, ref_latents3]
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                current_image_vl_setting = images_vl_settings[i]
                current_ref_setting = ref_latents_settings[i]

                # 处理 images_vl - 视觉编码
                if current_image_vl_setting == "ini(384x384)":
                    # 使用默认缩放逻辑
                    total = int(384 * 384)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                else:  # NaN - 不缩放，直接使用原始尺寸
                    s = samples
                images_vl.append(s.movedim(1, -1))

                # 处理 ref_latents - 潜在空间
                if vae is not None:
                    if current_ref_setting == "ini(1024*1024)":
                        # 使用默认缩放逻辑
                        total = int(1024 * 1024)
                        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                        width = round(samples.shape[3] * scale_by / 8.0) * 8
                        height = round(samples.shape[2] * scale_by / 8.0) * 8
                        s_ref = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    else:  # NaN - 不缩放，直接使用原始尺寸（仅调整为8的倍数）
                        # 确保尺寸是8的倍数以兼容VAE，但保持原始图像的实际大小
                        # 对于1248这样已经是8的倍数的尺寸，不会改变
                        width = (samples.shape[3] // 8) * 8
                        height = (samples.shape[2] // 8) * 8
                        # 只有当尺寸需要调整时才进行缩放
                        if width != samples.shape[3] or height != samples.shape[2]:
                            s_ref = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                        else:
                            s_ref = samples  # 尺寸已经是8的倍数，不需要缩放
                    
                    # 生成latent
                    latent = vae.encode(s_ref.movedim(1, -1)[:, :, :, :3])
                    ref_latents.append(latent)
                    
                    # 只有当这是用户选择的图像时，才保存latent用于输出
                    if vae is not None and i == chosen_latent_index:
                        chosen_latent = latent

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        # 处理latent输出
        output_latent = None
        if chosen_latent is not None:
            # 包装成ComfyUI的latent格式
            output_latent = {
                "samples": chosen_latent
            }
        else:
            # 如果没有生成latent（例如没有提供对应图像或vae），创建空的latent输出
            # 这样可以确保即使无法生成latent，节点也能正常运行
            output_latent = {
                "samples": torch.zeros([1, 4, 64, 64], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            }
        
        # 创建con_zero_out输出 - 零化pooled_output但保留reference_latents
        conditioning_zeroout = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            
            # 保留reference_latents信息
            n = [torch.zeros_like(t[0]), d]
            conditioning_zeroout.append(n)
        
        return (conditioning, conditioning_zeroout, output_latent)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFL:QwenEditPlusFastuseV2": AFL_Qweneditplus_FastuseV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:QwenEditPlusFastuseV2": "AFL_Qweneditplus_FastuseV2"
}