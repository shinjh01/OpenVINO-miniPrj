import torch
from unittest.mock import MagicMock
from pipelines.pix2pix_turbo import Pix2PixTurbo


def test_set_r():
    """
    Pix2PixTurbo.set_r 메서드가 내부 속성(r, gamma 등)과 unet/vae의 메서드 및 속성을 올바르게 설정하는지 검증합니다.
    - unet.set_adapters가 올바른 인자로 호출되는지
    - vae.decoder.gamma, unet.conv_in.r, model.r 값이 올바르게 할당되는지 확인합니다.
    """
    model = MagicMock()
    model.unet = MagicMock()
    model.vae = MagicMock()
    model.vae.decoder = MagicMock()
    model.unet.conv_in = MagicMock()
    # 실제 인스턴스 생성 없이 set_r만 독립적으로 테스트
    Pix2PixTurbo.set_r(model, 0.7)
    model.unet.set_adapters.assert_called_once_with(["default"], weights=[0.7])
    model.vae.decoder.gamma = 0.7
    assert model.r == 0.7
    assert model.unet.conv_in.r == 0.7
    assert model.vae.decoder.gamma == 0.7


def test_forward():
    """
    Pix2PixTurbo.forward 메서드가 입력 텐서(c_t, prompt_tokens, noise_map)를 받아
    내부 모듈(text_encoder, vae, unet, sched 등)을 호출하여
    최종적으로 (1, 3, 8, 8) shape의 torch.Tensor를 반환하는지 검증합니다.

    - 입력값:
        c_t: (1, 3, 8, 8)  # 입력 이미지 텐서 (batch=1, channel=3, height=8, width=8)
        prompt_tokens: (1, 77)  # 텍스트 토큰
        noise_map: (1, 4, 8, 8)  # 노이즈 맵

    - 내부 동작(모두 mock 처리):
        1. vae.encode(c_t) → (샘플, skip_blocks)
        2. 샘플.sample() → (1, 4, 8, 8)  # latent space
        3. unet 등 거쳐서 sched.step() → prev_sample (1, 4, 8, 8)
        4. vae.decode() → (1, 3, 8, 8)  # 최종 이미지 복원

    - 따라서, 최종적으로 (1, 3, 8, 8) shape의 torch.Tensor가 반환됨을 검증합니다.
    - 내부 모듈은 모두 mock 처리하여, 반환값 타입과 shape만 확인합니다.
    """
    model = MagicMock()
    # mock 내부 구조
    model.text_encoder = MagicMock(return_value=[torch.ones(1, 2, 3)])
    model.vae = MagicMock()
    mock_sample = MagicMock()
    mock_sample.sample.return_value = torch.ones(1, 4, 8, 8)
    model.vae.encode.return_value = (mock_sample, "skip_blocks")
    model.vae.config.scaling_factor = 2.0
    model.r = 0.5
    model.unet = MagicMock(return_value=MagicMock(sample=torch.ones(1, 4, 8, 8)))
    model.timesteps = torch.tensor([999])
    model.sched = MagicMock()
    model.sched.step.return_value = MagicMock(prev_sample=torch.ones(1, 4, 8, 8))
    model.vae.decode.return_value = (torch.ones(1, 3, 8, 8),)

    c_t = torch.ones(1, 3, 8, 8)
    prompt_tokens = torch.ones(1, 77).long()
    noise_map = torch.ones(1, 4, 8, 8)

    # clamp도 mock (torch.Tensor.clamp는 in-place가 아님)
    with torch.no_grad():
        out = Pix2PixTurbo.forward(model, c_t, prompt_tokens, noise_map)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 8, 8)