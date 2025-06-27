from gaze_runner import GazeRunner


def main():
    GazeRunner()\
        .download_model()\
        .check_model()\
        .load_models()\
        .run_ui_by_webcam()


if __name__ == "__main__":
    main()
