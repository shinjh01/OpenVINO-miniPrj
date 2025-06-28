from io import BytesIO
import base64


class ImageUtils:
    """
        Image 유틸.
    """
    def __init__(self):
        pass

    def pil_image_to_data_uri(self, img, format="PNG"):
        """
        역할:
            PIL 이미지를 base64로 인코딩해 data URI 문자열로 변환합니다.
        사용:
            Gradio 등 웹에서 이미지 미리보기에 사용.
        """
        buffered = BytesIO()
        img.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"