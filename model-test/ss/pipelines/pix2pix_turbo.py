import torch
from transformers import CLIPTextModel
from autoencoders.vae_autoencoder import VaeAutoEncoder
from diffusers import AutoencoderKL, UNet2DConditionModel
import types
from pathlib import Path
from peft import LoraConfig
import requests
from tqdm import tqdm
import copy
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from pipelines.pix2pix_twinconv import TwinConv
from pipelines.custom_ddpm_scheduler import CustomDDPMScheduler
from constants import (CHECK_POINT_DIR,
                       MODEL_NAME,
                       MODEL_VARIANT,
                       MODEL_SUB_FOLDER_UNET,
                       MODEL_SUB_FOLDER_VAE,
                       MODEL_SUB_FOLDER_TEXT_ENCODER,
                       DEVICE,
                       PRE_TRAIN_EDGE,
                       PRE_TRAIN_EDGE_URI,
                       PRE_TRAIN_EDGE_FILE_NAME,
                       PRE_TRAIN_SKETCH,
                       PRE_TRAIN_SKETCH_URI,
                       PRE_TRAIN_SKETCH_FILE_NAME
                       )


class Pix2PixTurbo(torch.nn.Module):
    """
    역할:
        텍스트+스케치→이미지 변환을 위한 전체 모델 구조를 정의합니다.
    동작:
        텍스트 인코더, VAE, UNet, LoRA 어댑터 등 다양한 컴포넌트를 초기화하고, 체크포인트를 불러와 가중치를 적용합니다.
    forward:
        텍스트, 이미지, 노이즈를 받아 최종 이미지를 생성합니다.
    """
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder=CHECK_POINT_DIR, lora_rank_unet=8):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_NAME,
            subfolder=MODEL_SUB_FOLDER_TEXT_ENCODER,
            variant=MODEL_VARIANT
        ).cpu()
        self.sched = CustomDDPMScheduler().make_1step_sched()
        
        vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder=MODEL_SUB_FOLDER_VAE, variant=MODEL_VARIANT)
        vae.encoder.forward = types.MethodType(VaeAutoEncoder.vae_encoder_fwd, vae.encoder)
        vae.decoder.forward = types.MethodType(VaeAutoEncoder.vae_decoder_fwd, vae.decoder)
        vae.encode = types.MethodType(VaeAutoEncoder.vae_encode, vae)
        vae.decode = types.MethodType(VaeAutoEncoder.vae_decode, vae)
        vae._decode = types.MethodType(VaeAutoEncoder.vae__decode, vae)

        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cpu()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cpu()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cpu()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cpu()
        vae.decoder.ignore_skip = False

        unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder=MODEL_SUB_FOLDER_UNET, variant=MODEL_VARIANT)
        ckpt_folder = Path(ckpt_folder)

        if pretrained_name == PRE_TRAIN_EDGE:
            ckpt_folder.mkdir(exist_ok=True)
            outf = ckpt_folder / PRE_TRAIN_EDGE_FILE_NAME
            if not outf:
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(PRE_TRAIN_EDGE_URI, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                with open(outf, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"]
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"]
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == PRE_TRAIN_SKETCH:
            # download from url
            ckpt_folder.mkdir(exist_ok=True)
            outf = ckpt_folder / PRE_TRAIN_SKETCH_FILE_NAME
            if not outf.exists():
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(PRE_TRAIN_SKETCH_URI, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                with open(outf, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"]
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"]
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                if k not in _sd_vae:
                    continue
                _sd_vae[k] = sd["state_dict_vae"][k]

            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"]
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"]
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        # unet.enable_xformers_memory_efficient_attention()
        unet.to(DEVICE)
        vae.to(DEVICE)
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cpu").long()
        self.text_encoder.requires_grad_(False)

    def set_r(self, r):
        self.unet.set_adapters(["default"], weights=[r])
        set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
        self.r = r
        self.unet.conv_in.r = r
        self.vae.decoder.gamma = r

    def forward(self, c_t, prompt_tokens, noise_map):
        caption_enc = self.text_encoder(prompt_tokens)[0]
        # scale the lora weights based on the r value
        sample, current_down_blocks = self.vae.encode(c_t)
        encoded_control = sample.sample() * self.vae.config.scaling_factor
        # combine the input and noise
        unet_input = encoded_control * self.r + noise_map * (1 - self.r)

        unet_output = self.unet(
            unet_input,
            self.timesteps,
            encoder_hidden_states=caption_enc,
        ).sample
        x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
        return (self.vae.decode(x_denoised / self.vae.config.scaling_factor, current_down_blocks)[0]).clamp(-1, 1)
