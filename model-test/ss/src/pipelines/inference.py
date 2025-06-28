import torch
import torchvision.transforms.functional as F
import numpy as np
import gradio as gr
import openvino as ov
import gc
from transformers import AutoTokenizer
from PIL import Image
from pathlib import Path
from pipelines.pix2pix_turbo import Pix2PixTurbo
from ui.image_utils import ImageUtils
from constants import (MODEL_NAME, MODEL_SUB_FOLDER_TOKENIZER, PRE_TRAIN_SKETCH, MODEL_XML_PATH)


class Inference:
    """
    Inference는 토크나이저 다운과 모델 컴파일을 초기화시 시작하므로
    싱글턴으로 만들어 초기화를 1번만 하도록 강제한다.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        tokenizer 생성 및 모델 컴파일
        """
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder=MODEL_SUB_FOLDER_TOKENIZER)
        self.compiled_model = self.compile_model()
        self.image_utils = ImageUtils()

    def compile_model(self, ov_model_path=Path(MODEL_XML_PATH), device="CPU"):
        """
        1. **모델 변환(컴파일)**
        - `ov_model_path`(예: pix2pix-turbo.xml) 파일이 없으면,
            - PyTorch 모델(`Pix2PixTurbo`)을 불러와서 eval 모드로 전환
            - 예시 입력(example_input)으로 OpenVINO 변환(ov.convert_model)
            - 변환된 모델을 XML(IR) 파일로 저장
        - 이미 변환된 모델이 있으면 변환 과정 생략

        2. **메모리 정리**
        - 변환이 끝난 후 `del pt_model`로 PyTorch 모델 인스턴스를 삭제
        - `gc.collect()`로 파이썬의 가비지 컬렉터를 강제로 실행해 메모리 즉시 회수 시도

        3. **OpenVINO 모델 컴파일**
        - 변환된 IR 모델(XML)을 OpenVINO로 로드 및 컴파일해서 반환
        """
        # 1. **모델 변환(컴파일)**
        pt_model = None

        if not ov_model_path.exists():
            pt_model = Pix2PixTurbo(PRE_TRAIN_SKETCH)
            pt_model.set_r(0.4)
            pt_model.eval()

        if not ov_model_path.exists():
            example_input = [torch.ones((1, 3, 512, 512)),
                             torch.ones([1, 77], dtype=torch.int64),
                             torch.ones([1, 4, 64, 64])
                             ]
            with torch.no_grad():
                ov_model = ov.convert_model(pt_model,
                                            example_input=example_input,
                                            input=[[1, 3, 512, 512], [1, 77], [1, 4, 64, 64]]
                                            )
                ov.save_model(ov_model, ov_model_path)
            del ov_model
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()
        # 2. **메모리 정리**
        del pt_model
        gc.collect()
        # 3. **OpenVINO 모델 컴파일**
        core = ov.Core()
        return core.compile_model(ov_model_path, device)

    def tokenize_prompt(self, prompt):
        """
        역할:
            입력된 텍스트 프롬프트를 토크나이저로 토큰화하여 텐서로 변환합니다.
        사용:
            텍스트를 모델 입력에 맞는 숫자 시퀀스로 바꿔줍니다.
        """
        caption_tokens = self.tokenizer(prompt,
                                        max_length=self.tokenizer.model_max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt"
                                        ).input_ids
        return caption_tokens

    def run(self, image, prompt, prompt_template, style_name, seed):
        """
        역할:
            Gradio 인터페이스에서 실제로 이미지를 생성하는 함수입니다.
        동작:
            입력 스케치와 프롬프트를 받아, 토큰화 → 모델 추론 → 결과 이미지를 반환합니다.
        """
        print(f"prompt: {prompt}")
        print("sketch updated")
        if image is None:
            ones = Image.new("L", (512, 512), 255)
            temp_uri = self.image_utils.pil_image_to_data_uri(ones)
            return ones, gr.update(link=temp_uri), gr.update(link=temp_uri)
        prompt = prompt_template.replace("{prompt}", prompt)
        image = image.convert("RGB")
        image_t = F.to_tensor(image) > 0.5
        print(f"seed={seed}")
        caption_tokens = self.tokenizer(prompt,
                                        max_length=self.tokenizer.model_max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt"
                                        ).input_ids.cpu()
        with torch.no_grad():
            c_t = image_t.unsqueeze(0)
            torch.manual_seed(seed)
            B, C, H, W = c_t.shape
            noise = torch.randn((1, 4, H // 8, W // 8))
            output_image = torch.from_numpy(self.compiled_model([c_t.to(torch.float32), caption_tokens, noise])[0])
        output_pil = F.to_pil_image(output_image[0].cpu() * 0.5 + 0.5)
        input_sketch_uri = self.image_utils.pil_image_to_data_uri(Image.fromarray(255 - np.array(image)))
        output_image_uri = self.image_utils.pil_image_to_data_uri(output_pil)
        return (
            output_pil,
            gr.update(link=input_sketch_uri),
            gr.update(link=output_image_uri),
        )
