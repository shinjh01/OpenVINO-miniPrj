from dataclasses import dataclass


@dataclass
class GazeModels:
    """
    로드 된 모델에 대한 VO
    """
    gaze: object = None
    face: object = None
    landmarks: object = None
    head_pose: object = None
