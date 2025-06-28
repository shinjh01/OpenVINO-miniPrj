from unittest.mock import patch, MagicMock
from pipelines.custom_ddpm_scheduler import CustomDDPMScheduler


@patch("pipelines.custom_ddpm_scheduler.DDPMScheduler")
def test_make_1step_sched(mock_ddpm_scheduler):
    """
    CustomDDPMScheduler.make_1step_sched 메서드가
    DDPMScheduler를 from_pretrained으로 생성하고,
    set_timesteps(1, device="cpu")를 호출하며,
    alphas_cumprod.cpu()를 호출한 뒤,
    최종적으로 mock된 scheduler 인스턴스를 반환하는지 검증합니다.

    - mock_ddpm_scheduler.from_pretrained이 한 번 호출되는지
    - mock_sched_instance.set_timesteps가 1, device="cpu"로 호출되는지
    - mock_alphas.cpu()가 한 번 호출되는지
    - 반환값이 mock_sched_instance인지 확인합니다.
    """
    mock_sched_instance = MagicMock()
    mock_alphas = MagicMock()
    mock_alphas.cpu = MagicMock(return_value="cpu_tensor")
    mock_sched_instance.alphas_cumprod = mock_alphas
    mock_ddpm_scheduler.from_pretrained.return_value = mock_sched_instance

    # 테스트 실행
    scheduler = CustomDDPMScheduler()
    result = scheduler.make_1step_sched()

    # from_pretrained이 올바르게 호출되었는지 확인
    mock_ddpm_scheduler.from_pretrained.assert_called_once()
    # set_timesteps가 1로 호출되었는지 확인
    mock_sched_instance.set_timesteps.assert_called_once_with(1, device="cpu")
    # alphas_cumprod.cpu()가 호출되었는지 확인 (mock_alphas를 직접 체크)
    mock_alphas.cpu.assert_called_once()
    # 반환값이 mock 객체인지 확인
    assert result == mock_sched_instance