from pathlib import Path
import subprocess
from constants import (
    MODEL_DIR,
    GAZE_MODEL,
    FACE_DETECTION_MODEL,
    LANDMARKS_MODEL,
    HEAD_POSE_MODEL,
    MODEL_INTEL_DIR
)


class ModelDownload:
    """
    모델 다운로드
    """

    def run_omz_download(self, model_name: str, output_dir: str = MODEL_DIR):
        """
        omz_downloader를 사용해 OpenVINO 모델을 다운로드합니다.

        Args:
            model_name (str): 다운로드할 모델 이름
            output_dir (str): 모델 저장 경로

        Returns:
            None
        """
        model_path = Path(MODEL_INTEL_DIR) / model_name
        if model_path.exists():
            print(f"{model_path} 이미 존재합니다.")
            return

        cmd = [
            "omz_downloader",
            "--name", model_name,
            "--output_dir", output_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)

        if result.returncode != 0:
            print("모델 다운로드 실패:", result.stderr)
        else:
            print("모델 다운로드 완료:", model_name)

    def run(self):
        # 모델 저장 디렉토리 생성
        model_dir = Path(MODEL_DIR)

        model_dir.mkdir(exist_ok=True)

        # omz_downloader 도구를 사용하여 모델 다운로드
        self.run_omz_download(GAZE_MODEL)

        # 시선 추정을 위해 필요한 추가 모델들 다운로드
        # 얼굴 검출 모델, 얼굴 랜드마크 모델, 머리 자세 추정 모델
        for model_name in [FACE_DETECTION_MODEL, LANDMARKS_MODEL, HEAD_POSE_MODEL]:
            print(f"다운로드 중: {model_name}")
            self.run_omz_download(model_name)

        print("\n모든 모델 다운로드 완료!")
