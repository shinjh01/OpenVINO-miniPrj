from unittest.mock import MagicMock, patch
from pipelines.inference import Inference
from PIL import Image
import numpy as np
import torch


@patch("pipelines.inference.Inference.compile_model")
@patch("pipelines.inference.AutoTokenizer")
@patch("pipelines.inference.ImageUtils")
def test_singleton(mock_image_utils, mock_tokenizer, mock_compile_model):
    """
    Inference 클래스가 싱글턴 패턴으로 동작하는지 검증합니다.

    - Inference()를 여러 번 호출해도 항상 같은 인스턴스(a is b)가 반환되어야 합니다.
    - 내부적으로 모델 컴파일 등 실제 연산은 mock 처리하여, 인스턴스 생성만 검증합니다.
    """
    mock_compile_model.return_value = MagicMock()
    a = Inference()
    b = Inference()
    assert a is b  # 싱글턴 패턴 확인


@patch("pipelines.inference.AutoTokenizer")
@patch("pipelines.inference.Inference.compile_model")
@patch("pipelines.inference.ImageUtils")
def test_tokenize_prompt(mock_image_utils, mock_compile_model, mock_tokenizer):
    """
    Inference.tokenize_prompt 메서드가 입력 프롬프트 문자열을
    내부 토크나이저(mock)로 토큰화하여, input_ids 값을 올바르게 반환하는지 검증합니다.

    - mock_tokenizer는 model_max_length 속성을 가지고,
      __call__ 시 input_ids="token_ids"를 반환하도록 설정되어 있습니다.

    - 입력값:
        "hello world" (임의의 프롬프트)

    - 반환값:
        "token_ids" (mock된 토크나이저의 input_ids)

    - 즉, 실제 토크나이저 동작 대신 mock을 통해
      함수가 input_ids를 올바르게 추출해 반환하는지만 검증합니다.
    """
    mock_tokenizer.from_pretrained.return_value = MagicMock(model_max_length=77)
    inf = Inference()
    inf.tokenizer = MagicMock()
    inf.tokenizer.model_max_length = 77
    inf.tokenizer.return_value = MagicMock(input_ids="token_ids")
    inf.tokenizer.__call__ = MagicMock(return_value=MagicMock(input_ids="token_ids"))

    result = inf.tokenize_prompt("hello world")
    assert result == "token_ids"


@patch("pipelines.inference.F")
@patch("pipelines.inference.gr")
@patch("pipelines.inference.ImageUtils")
@patch("pipelines.inference.Inference.compile_model")
@patch("pipelines.inference.AutoTokenizer")
def test_run_with_image(
    mock_tokenizer, mock_compile_model, mock_image_utils, mock_gr, mock_F
):
    """
    Inference.run 메서드가 입력 이미지(pil_img), 프롬프트, 스타일, 시드 등을 받아
    내부적으로 토크나이즈, 텐서 변환, 모델 추론, 결과 이미지 변환, data_uri 변환 등을 거쳐
    (PIL.Image, dict, dict) 튜플을 반환하는지 검증합니다.

    - 입력값:
        pil_img: (512, 512) RGB PIL 이미지
        prompt: str
        prompt_template: str
        style: str
        seed: int

    - 내부 동작(모두 mock 처리):
        1. F.to_tensor(pil_img) → torch.Tensor (3, 512, 512)
        2. compiled_model 등 모델 추론 → np.ones((3, 512, 512))
        3. F.to_pil_image → PIL.Image
        4. image_utils.pil_image_to_data_uri → "dummy_uri"
        5. gr.update → dict

    - 반환값:
        result[0]: PIL.Image.Image (생성된 결과 이미지)
        result[1]: dict ({"link": "dummy_uri"})
        result[2]: dict ({"link": "dummy_uri"})

    - 각 반환값의 타입과 내부 값이 예상대로 나오는지 검증합니다.
    """
    # 준비
    inf = Inference()
    inf.compiled_model = MagicMock(return_value=[np.ones((3, 512, 512), dtype=np.float32)])
    inf.tokenizer = MagicMock()
    inf.tokenizer.model_max_length = 77
    inf.tokenizer.return_value = MagicMock(input_ids=MagicMock(cpu=MagicMock(return_value="caption_tokens")))
    inf.image_utils.pil_image_to_data_uri = MagicMock(return_value="dummy_uri")
    mock_F.to_tensor.return_value = torch.ones(3, 512, 512)
    mock_F.to_pil_image.return_value = Image.new("RGB", (512, 512))

    # 입력 이미지
    pil_img = Image.new("RGB", (512, 512))
    # gr.update mock
    mock_gr.update.side_effect = lambda **kwargs: kwargs

    # 실행
    result = inf.run(pil_img, "prompt", "{prompt}", "style", 42)

    # 반환값 체크
    assert isinstance(result, tuple)
    assert isinstance(result[0], Image.Image)
    assert isinstance(result[1], dict)
    assert isinstance(result[2], dict)
    assert result[1]["link"] == "dummy_uri"
    assert result[2]["link"] == "dummy_uri"


@patch("pipelines.inference.F")
@patch("pipelines.inference.gr")
@patch("pipelines.inference.ImageUtils")
@patch("pipelines.inference.Inference.compile_model")
@patch("pipelines.inference.AutoTokenizer")
def test_run_with_none_image(
    mock_tokenizer, mock_compile_model, mock_image_utils, mock_gr, mock_F
):
    """
    Inference.run 메서드에 입력 이미지가 None으로 들어올 때의 동작을 검증합니다.

    - inf.run(None, ...) 호출 시 내부에서
      1. 빈 흰색 PIL 이미지(512x512, RGB)가 생성되어 반환됩니다.
      2. image_utils.pil_image_to_data_uri가 두 번 호출되어 각각 dummy_uri를 반환합니다.
      3. gr.update는 dict로 mock 처리되어 반환됩니다.

    - 반환값:
      result[0]: PIL.Image.Image (흰색 512x512)
      result[1]: dict ({"link": "dummy_uri"})
      result[2]: dict ({"link": "dummy_uri"})

    - 각 반환값의 타입과 내부 값이 예상대로 나오는지 검증합니다.
    """
    inf = Inference()
    inf.image_utils.pil_image_to_data_uri = MagicMock(return_value="dummy_uri")
    mock_gr.update.side_effect = lambda **kwargs: kwargs

    result = inf.run(None, "prompt", "{prompt}", "style", 42)
    from PIL import Image as PILImage
    assert isinstance(result[0], PILImage.Image)
    assert isinstance(result[1], dict)
    assert isinstance(result[2], dict)
    assert result[1]["link"] == "dummy_uri"
    assert result[2]["link"] == "dummy_uri"
