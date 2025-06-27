from gaze_model.model_compile import ModelCompile
from gaze_model.model_download import ModelDownload
from ui.draw_gaze import DrawGaze


class GazeRunner:
    """
        실행을 쉽게하기 위한 클래스 (빌더 패턴)
    """
    def __init__(self):
        self.model_download = ModelDownload()
        self.model_compile = ModelCompile()
        self.draw_gaze = DrawGaze()
        self.gaze_models = None

    def download_model(self):
        self.model_download.run()
        return self

    def check_model(self):
        self.model_compile.check_models()
        return self

    def load_models(self):
        self.gaze_models = self.model_compile.load_models()
        return self

    def run_ui_by_webcam(self):
        self.draw_gaze.draw_gaze_by_webcam(self.gaze_models)
        return self
