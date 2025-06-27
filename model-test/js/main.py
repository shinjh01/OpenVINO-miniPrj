from graze_model.model_compile import ModelCompile
from graze_model.model_download import ModelDownload
from ui.draw_gaze import DrawGaze


def main():
    md = ModelDownload()
    md.run()
    mc = ModelCompile()
    mc.check_models()
    (gaze_compiled, face_compiled, landmarks_compiled, head_pose_compiled) = mc.load_models()

    dg = DrawGaze()
    dg.draw_gaze(gaze_compiled, face_compiled, landmarks_compiled, head_pose_compiled)


if __name__ == "__main__":
    main()
