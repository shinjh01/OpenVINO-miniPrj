from ui.gradio_helper import GradioHelper
from pipelines.inference import Inference


def main():
    infernce = Inference()

    demo = GradioHelper().make_demo(fn=infernce.run)

    try:
        demo.queue().launch(debug=True)
    except Exception:
        demo.queue().launch(debug=True, share=True)


if __name__ == "__main__":
    main()
