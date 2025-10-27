class AFL_Fast_Text:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "text": ("STRING", {"multiline": True}),
            },}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'text_box_node'
    CATEGORY = "AFL/实用工具"

    def text_box_node(self, text):
        return (text,)

NODE_CLASS_MAPPINGS = {
    "AFL:Fast_Text": AFL_Fast_Text
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:Fast_Text": "AFL_fast_text"
}