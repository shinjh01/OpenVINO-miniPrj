from graze_model.model_compile import ModelCompile
from ui.draw_gaze import DrawGaze


def main():
    print("Hello, OpenVINO-turtleNeck!")
    mc = ModelCompile()
    mc.check_models()
    mc.load_models()

    dg = DrawGaze()
    dg.draw_gaze()


if __name__ == "__main__":
    main()
