import torch
from unittest.mock import MagicMock, patch
from pipelines.pix2pix_twinconv import TwinConv


@patch("pipelines.pix2pix_twinconv.copy.deepcopy", side_effect=lambda x: x)
def test_twinconv_forward(mock_deepcopy):
    """
    이 테스트케이스는 `TwinConv` 클래스의 `forward` 메서드가 **정상적으로 동작하는지** 검증합니다.

        ### 주요 역할

        1. **copy.deepcopy 패치**
        - `@patch("pipelines.pix2pix_twinconv.copy.deepcopy", side_effect=lambda x: x)`  
            → 생성자에서 deepcopy를 해도 실제로는 원본 객체(`conv1`, `conv2`)가 그대로 사용되도록 만듭니다.

        2. **Mock conv 모듈 준비**
        - `conv1`, `conv2`를 `MagicMock`으로 생성  
        - 각각 입력 텐서 `x`를 받으면,  
            - `conv1`은 값이 2인 텐서  
            - `conv2`는 값이 5인 텐서를 반환하도록 설정

        3. **TwinConv 인스턴스 생성 및 r값 설정**
        - `model = TwinConv(conv1, conv2)`
        - `model.r = 0.3` (가중치 비율)

        4. **forward 연산 및 결과 검증**
        - `out = model.forward(x)` 실행  
        - 결과는 `2*(1-0.3) + 5*0.3 = 2.9`가 되어야 하므로  
            - `assert torch.allclose(out, torch.ones_like(x) * 2.9)`로 값 검증

        5. **Mock 호출 검증**
        - `conv1`과 `conv2`가 각각 한 번씩 `x`로 호출되었는지 확인

        ---

        ### 요약

        - **TwinConv의 선형 조합 로직**이 r값에 따라 올바르게 동작하는지 확인
        - 내부에 전달된 conv 모듈이 실제로 호출되는지 확인
        - deepcopy로 인한 mock 깨짐 현상을 방지

    """
    conv1 = MagicMock()
    conv2 = MagicMock()
    x = torch.ones(1, 3, 8, 8)
    conv1_tensor = torch.ones_like(x) * 2
    conv1.return_value = conv1_tensor
    conv2.return_value = torch.ones_like(x) * 5

    model = TwinConv(conv1, conv2)
    model.r = 0.3

    out = model.forward(x)
    assert torch.allclose(out, torch.ones_like(x) * 2.9)
    conv1.assert_called_once_with(x)
    conv2.assert_called_once_with(x)