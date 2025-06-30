from PIL import Image
from ui.image_utils import ImageUtils


def test_pil_image_to_data_uri_png():
    """
    ImageUtils.pil_image_to_data_uri 함수의 PNG 변환 테스트입니다.

    입력:
        - PIL.Image 객체 (10x10, RGB, 빨간색)
        - format="PNG"
    출력:
        - "data:image/png;base64,"로 시작하는 문자열
        - base64로 인코딩된 PNG 이미지 데이터가 포함되어 있음
    예시:
        >>> img = Image.new("RGB", (10, 10), color="red")
        >>> utils = ImageUtils()
        >>> data_uri = utils.pil_image_to_data_uri(img, format="PNG")
        >>> data_uri.startswith("data:image/png;base64,")
        True
        >>> isinstance(data_uri, str)
        True
        >>> len(data_uri) > len("data:image/png;base64,")
        True
    설명:
        이 테스트는 PIL 이미지 객체를 PNG 포맷의 data URI 문자열로 변환하는 기능을 검증합니다.
        반환된 문자열이 올바른 접두사와 비어있지 않은 base64 데이터를 포함하는지 확인합니다.
    """
    img = Image.new("RGB", (10, 10), color="red")
    utils = ImageUtils()
    data_uri = utils.pil_image_to_data_uri(img, format="PNG")
    assert data_uri.startswith("data:image/png;base64,")
    assert isinstance(data_uri, str)
    # base64 부분이 비어있지 않은지 확인
    assert len(data_uri) > len("data:image/png;base64,")


def test_pil_image_to_data_uri_jpeg():
    """
    ImageUtils.pil_image_to_data_uri 함수의 JPEG 변환 테스트입니다.

    입력:
        - PIL.Image 객체 (10x10, RGB, 파란색)
        - format="JPEG"
    출력:
        - "data:image/jpeg;base64,"로 시작하는 문자열
        - base64로 인코딩된 JPEG 이미지 데이터가 포함되어 있음
    예시:
        >>> img = Image.new("RGB", (10, 10), color="blue")
        >>> utils = ImageUtils()
        >>> data_uri = utils.pil_image_to_data_uri(img, format="JPEG")
        >>> data_uri.startswith("data:image/jpeg;base64,")
        True
        >>> isinstance(data_uri, str)
        True
        >>> len(data_uri) > len("data:image/jpeg;base64,")
        True
    설명:
        이 테스트는 PIL 이미지 객체를 JPEG 포맷의 data URI 문자열로 변환하는 기능을 검증합니다.
        반환된 문자열이 올바른 접두사와 비어있지 않은 base64 데이터를 포함하는지 확인합니다.
    """
    img = Image.new("RGB", (10, 10), color="blue")
    utils = ImageUtils()
    data_uri = utils.pil_image_to_data_uri(img, format="JPEG")
    assert data_uri.startswith("data:image/jpeg;base64,")
    assert isinstance(data_uri, str)
    assert len(data_uri) > len("data:image/jpeg;base64,")