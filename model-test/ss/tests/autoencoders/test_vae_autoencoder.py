import torch
from unittest.mock import MagicMock
from autoencoders.vae_autoencoder import VaeAutoEncoder


class DummyEncoder(torch.nn.Module):
    def forward(self, x):
        return x + 1, ["skip1", "skip2"]


class DummyQuantConv(torch.nn.Module):
    def forward(self, h):
        return h * 2


class DummyPostQuantConv(torch.nn.Module):
    def forward(self, z):
        return z + 10


class DummyDecoder(torch.nn.Module):
    def forward(self, z, skip_acts):
        return z - 1


class DummySelf:
    def __init__(self):
        self.encoder = DummyEncoder()
        self.quant_conv = DummyQuantConv()
        self.post_quant_conv = DummyPostQuantConv()
        self.decoder = DummyDecoder()


class DummyUpBlocks(list):
    def parameters(self):
        return iter([torch.zeros(1)])


def test_vae_encode_returns_posterior_and_down_blocks():
    """
    VaeAutoEncoder.vae_encode 메서드가
    입력 x와 mock된 encoder, quant_conv, post_quant_conv, decoder를 받아
    (posterior, down_blocks) 튜플을 반환하는지 검증합니다.

    - dummy.encoder는 (x+1, ["skip1", "skip2"])를 반환하도록 구현되어 있습니다.
    - 입력값:
        x: torch.randn(1, 3, 8, 8)

    - 반환값:
        posterior: DiagonalGaussianDistribution (diffusers에서 제공)
        down_blocks: list

    - 즉, encoder와 quant/post_quant_conv가 올바르게 호출되고,
      posterior와 down_blocks의 타입이 예상대로 나오는지 확인합니다.
    """
    dummy = DummySelf()
    x = torch.randn(1, 3, 8, 8)
    posterior, down_blocks = VaeAutoEncoder.vae_encode(dummy, x)
    # posterior는 DiagonalGaussianDistribution, down_blocks는 리스트
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    assert isinstance(posterior, DiagonalGaussianDistribution)
    assert isinstance(down_blocks, list)


def test_vae_decode_returns_tuple():
    """
    VaeAutoEncoder.vae_decode 메서드가
    입력 z, skip_acts와 mock된 _decode 메서드를 받아
    (Tensor,) 형태의 튜플을 반환하는지 검증합니다.

    - dummy._decode는 항상 (torch.ones(1, 3, 8, 8),)을 반환하도록 mock 처리되어 있습니다.
    - 입력값:
        z: torch.randn(1, 3, 8, 8)
        skip_acts: ["skip1", "skip2"]

    - 반환값:
        result: tuple, result[0] == torch.ones(1, 3, 8, 8)

    - 즉, _decode가 올바르게 호출되고, 반환값이 튜플이며
      내부 텐서 값이 예상대로 나오는지 확인합니다.
    """
    dummy = DummySelf()
    z = torch.randn(1, 3, 8, 8)
    skip_acts = ["skip1", "skip2"]
    # _decode를 모킹
    dummy._decode = MagicMock(return_value=(torch.ones(1, 3, 8, 8),))
    result = VaeAutoEncoder.vae_decode(dummy, z, skip_acts)
    assert isinstance(result, tuple)
    assert torch.equal(result[0], torch.ones(1, 3, 8, 8))


def test_vae__decode_returns_tuple():
    """
    VaeAutoEncoder.vae__decode 메서드가
    입력 z, skip_acts와 mock된 decoder를 받아
    (Tensor,) 형태의 튜플을 반환하는지 검증합니다.

    - dummy.decoder는 항상 torch.ones(1, 3, 8, 8)을 반환하도록 mock 처리되어 있습니다.
    - 입력값:
        z: torch.randn(1, 3, 8, 8)
        skip_acts: ["skip1", "skip2"]

    - 반환값:
        result: tuple, result[0] == torch.ones(1, 3, 8, 8)

    - 즉, decoder가 올바르게 호출되고, 반환값이 튜플이며
      내부 텐서 값이 예상대로 나오는지 확인합니다.
    """
    dummy = DummySelf()
    z = torch.randn(1, 3, 8, 8)
    skip_acts = ["skip1", "skip2"]
    # decoder를 모킹
    dummy.decoder = MagicMock(return_value=torch.ones(1, 3, 8, 8))
    result = VaeAutoEncoder.vae__decode(dummy, z, skip_acts)
    assert isinstance(result, tuple)
    assert torch.equal(result[0], torch.ones(1, 3, 8, 8))


def test_vae_encoder_fwd_and_decoder_fwd_shapes():
    """
    VaeAutoEncoder.vae_encoder_fwd와 vae_decoder_fwd 메서드가
    입력 텐서와 더미 모듈(mock/self)로 호출될 때
    각각 올바른 타입의 결과(encoder: (Tensor, list), decoder: Tensor)를 반환하는지 검증합니다.

    - 인코더 fwd:
        - 입력: sample (torch.zeros(1, 3, 8, 8))
        - 내부적으로 conv_in, down_blocks, mid_block, conv_norm_out, conv_act, conv_out 등이 순차적으로 호출됨
        - 반환: (out, skips) (out은 Tensor, skips는 list)

    - 디코더 fwd:
        - 입력: sample, ["skip1", "skip2"]
        - 내부적으로 up_blocks, mid_block, conv_norm_out, conv_act, conv_out 등이 순차적으로 호출됨
        - 반환: out2 (Tensor)

    - 모든 내부 모듈은 MagicMock/더미로 대체하여, 타입과 호출 흐름만 검증합니다.
    """
    # 인코더 fwd
    dummy_self = MagicMock()
    dummy_self.conv_in = MagicMock(side_effect=lambda x: x + 1)
    dummy_self.down_blocks = [MagicMock(side_effect=lambda x: x + 1) for _ in range(2)]
    dummy_self.mid_block = MagicMock(side_effect=lambda x: x + 1)
    dummy_self.conv_norm_out = MagicMock(side_effect=lambda x: x + 1)
    dummy_self.conv_act = MagicMock(side_effect=lambda x: x + 1)
    dummy_self.conv_out = MagicMock(side_effect=lambda x: x + 1)
    sample = torch.zeros(1, 3, 8, 8)
    out, skips = VaeAutoEncoder.vae_encoder_fwd(dummy_self, sample)
    assert isinstance(out, torch.Tensor)
    assert isinstance(skips, list)

    # 디코더 fwd
    dummy_self.up_blocks = DummyUpBlocks([MagicMock(side_effect=lambda x, y=None: x + 1) for _ in range(2)])
    dummy_self.mid_block = MagicMock(side_effect=lambda x, y=None: x + 1)
    dummy_self.conv_norm_out = MagicMock(side_effect=lambda x, y=None: x + 1)
    dummy_self.conv_act = MagicMock(side_effect=lambda x: x + 1)
    dummy_self.conv_out = MagicMock(side_effect=lambda x: x + 1)
    dummy_self.ignore_skip = True
    out2 = VaeAutoEncoder.vae_decoder_fwd(dummy_self, sample, ["skip1", "skip2"])
    assert isinstance(out2, torch.Tensor)