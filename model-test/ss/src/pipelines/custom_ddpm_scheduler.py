from diffusers import DDPMScheduler
from constants import (MODEL_NAME, DEVICE, MODEL_SUB_FOLDER_SCHEDULER)


class CustomDDPMScheduler:
    def make_1step_sched(self):
        """
        이 코드는 Stable Diffusion Turbo에서 사용하는 **노이즈 스케줄러(DDPMScheduler)**를
        "한 스텝만" 동작하도록 설정하는 함수입니다.
        """
        noise_scheduler_1step = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder=MODEL_SUB_FOLDER_SCHEDULER)
        noise_scheduler_1step.set_timesteps(1, device=DEVICE)
        noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cpu()
        return noise_scheduler_1step
