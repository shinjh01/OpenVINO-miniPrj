import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class VaeAutoEncoder:

    def _vae_encoder_fwd(self, sample):
        """
        역할:
            VAE(Variational AutoEncoder) 인코더의 forward 연산을 커스텀하게 정의합니다.
        동작:
            입력 이미지를 여러 블록을 거쳐 인코딩하고, 중간 결과(스킵 연결용)도 함께 반환합니다.
        사용:
            VAE 인코더의 내부 연산을 오버라이드할 때 사용.
        """
        sample = self.conv_in(sample)
        l_blocks = []
        # down
        for down_block in self.down_blocks:
            l_blocks.append(sample)
            sample = down_block(sample)
        # middle
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        current_down_blocks = l_blocks
        return sample, current_down_blocks

    def _vae_decoder_fwd(self, sample, incoming_skip_acts, latent_embeds=None):
        """
        역할:
            VAE 디코더의 forward 연산을 커스텀하게 정의합니다.
        동작:
            인코딩된 latent와 스킵 연결을 받아, 여러 업샘플링 블록을 거쳐 이미지를 복원합니다.
        사용:
            VAE 디코더의 내부 연산을 오버라이드할 때 사용.    
        """
        sample = self.conv_in(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)
        if not self.ignore_skip:
            skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
            # up
            for idx, up_block in enumerate(self.up_blocks):
                skip_in = skip_convs[idx](incoming_skip_acts[::-1][idx] * self.gamma)
                # add skip
                sample = sample + skip_in
                sample = up_block(sample, latent_embeds)
        else:
            for idx, up_block in enumerate(self.up_blocks):
                sample = up_block(sample, latent_embeds)
        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample

    def vae_encode(self, x: torch.FloatTensor, *args, **kwargs):
        """
        역할:
            이미지를 latent(잠재공간) 표현으로 인코딩합니다.
        동작:
            인코더를 통해 이미지를 잠재공간으로 변환하고, 분포 객체와 스킵 연결 결과를 반환합니다.
        사용:
            이미지 → latent 변환.

        Args:
            x (`torch.FloatTensor`): Input batch of images.

        Returns:
            The latent representations of the encoded images. If `return_dict` is True, a
            [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        h, down_blocks = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        return (posterior, down_blocks)

    def vae_decode(self, z: torch.FloatTensor, skip_acts):
        """
        역할:
            latent(잠재공간) 표현을 이미지로 복원합니다.
        동작:
            내부적으로 _decode를 호출해 복원 이미지를 반환합니다.
        사용:
            latent → 이미지 변환.
        """
        decoded = self._decode(z, skip_acts)[0]
        return (decoded,)

    def vae__decode(self, z: torch.FloatTensor, skip_acts):
        """
        역할:
            latent를 post-quantization 처리 후 디코더로 복원합니다.
        동작:
            post_quant_conv를 거쳐 디코더로 이미지를 복원합니다.
        사용:
            latent → 이미지 변환(특정 구조에서 사용).
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z, skip_acts)

        return (dec,)