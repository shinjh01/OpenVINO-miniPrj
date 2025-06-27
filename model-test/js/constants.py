# 모델 저장 경로
MODEL_DIR = "./model-test/js/models"
MODEL_INTEL_DIR = "./model-test/js/models/intel"

# 시선 추정 모델 다운로드
GAZE_MODEL = "gaze-estimation-adas-0002"
GAZE_MODEL_XML = "gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"

# 1. 얼굴 검출 모델
FACE_DETECTION_MODEL = "face-detection-adas-0001"
FACE_DETECTION_MODEL_XML = "face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
# 2. 얼굴 랜드마크 모델
LANDMARKS_MODEL = "landmarks-regression-retail-0009"
LANDMARKS_MODEL_XML = "landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
# 3. 머리 자세 추정 모델
HEAD_POSE_MODEL = "head-pose-estimation-adas-0001"
HEAD_POSE_MODEL_XML = "head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"

DEVICE_CPU = "CPU"  # GPU가 있으면 "GPU"로 변경 가능
