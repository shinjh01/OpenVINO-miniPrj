# 모델 저장 경로
CHECK_POINT_DIR = "./model-test/ss/checkpoints"
MODEL_NAME = "stabilityai/sd-turbo"
MODEL_SUB_FOLDER_UNET = "unet"
MODEL_SUB_FOLDER_VAE = "vae"
MODEL_SUB_FOLDER_TEXT_ENCODER = "text_encoder"
MODEL_SUB_FOLDER_SCHEDULER = "scheduler"
MODEL_SUB_FOLDER_TOKENIZER = "tokenizer"
MODEL_XML_PATH = "./model-test/ss/model/pix2pix-turbo.xml"

MODEL_VARIANT = "fp16"

PRE_TRAIN_EDGE = "edge_to_image"
PRE_TRAIN_EDGE_URI = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
PRE_TRAIN_EDGE_FILE_NAME = "edge_to_image_loras.pkl"

PRE_TRAIN_SKETCH = "sketch_to_image_stochastic"
PRE_TRAIN_SKETCH_URI = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
PRE_TRAIN_SKETCH_FILE_NAME = "sketch_to_image_stochastic_lora.pkl"


DEVICE = "cpu"  # GPU가 있으면 "GPU"로 변경 가능
