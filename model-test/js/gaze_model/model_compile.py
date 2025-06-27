from pathlib import Path
import openvino as ov
from gaze_model.gaze_model_data import GazeModels
from constants import (
    GAZE_MODEL_XML,
    FACE_DETECTION_MODEL_XML,
    LANDMARKS_MODEL_XML,
    HEAD_POSE_MODEL_XML,
    MODEL_INTEL_DIR,
    DEVICE_CPU
)


class ModelCompile:
    """
    OpenVINO 모델 파일의 존재 여부를 확인하고, 모델을 메모리에 로드하는 클래스입니다.
    """
    def __init__(self):
        self.base_path = Path(MODEL_INTEL_DIR)
        self.gaze_path = self.base_path / GAZE_MODEL_XML
        self.face_detection_path = (self.base_path / FACE_DETECTION_MODEL_XML)
        self.landmarks_path = (self.base_path / LANDMARKS_MODEL_XML)
        self.head_pose_path = (self.base_path / HEAD_POSE_MODEL_XML)

    def check_models(self):
        """
        모델 파일들의 존재 여부를 출력합니다.
        """
        model_files = {
            "시선 추정": self.gaze_path,
            "얼굴 검출": self.face_detection_path,
            "랜드마크": self.landmarks_path,
            "머리 자세": self.head_pose_path
        }

        print("모델 파일 확인:")
        for name, path in model_files.items():
            if path.exists():
                print(f"✓ {name}: {path}")
            else:
                print(f"✗ {name}: {path} (파일이 없습니다)")

    def load_models(self) -> GazeModels:
        """
        모델 파일들을 로드합니다.
        """
        core = ov.Core()
        print(f"\nOpenVINO 버전: {core.get_versions(core.available_devices[0])}")
        print(f"사용 가능한 디바이스: {core.available_devices}")
      
        # 모델들을 메모리에 로드하기
        try:
            # 모델 읽기 및 컴파일
            gaze_model = core.read_model(model=self.gaze_path)
            face_model = core.read_model(model=self.face_detection_path)
            landmarks_model = core.read_model(model=self.landmarks_path)
            head_pose_model = core.read_model(model=self.head_pose_path)

            gaze_models = GazeModels()

            # 모델 컴파일 (추론 준비)
            gaze_models.gaze = core.compile_model(model=gaze_model, device_name=DEVICE_CPU)

            gaze_models.face = core.compile_model(model=face_model, device_name=DEVICE_CPU)

            gaze_models.landmarks = core.compile_model(model=landmarks_model, device_name=DEVICE_CPU)

            gaze_models.head_pose = core.compile_model(model=head_pose_model, device_name=DEVICE_CPU)

            print("✓ 모든 모델이 성공적으로 로드되었습니다!")

            # 모델 입출력 정보 확인
            print("\n=== 시선 추정 모델 정보 ===")
            print(f"입력: {[input.any_name for input in gaze_model.inputs]}")
            print(f"출력: {[output.any_name for output in gaze_model.outputs]}")

            print("\n=== 얼굴 검출 모델 정보 ===")
            print(f"입력: {[input.any_name for input in face_model.inputs]}")
            print(f"출력: {[output.any_name for output in face_model.outputs]}")

        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("모델 파일들이 올바르게 다운로드되었는지 확인해주세요.")

        return gaze_models
