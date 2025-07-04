from typing import Callable
import gradio as gr
from PIL import Image
import numpy as np


class GradioHelper:

    def __init__(self):
        self.style_list = [
            {
                "name": "Cinematic",
                "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
            },
            {
                "name": "3D Model",
                "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
            },
            {
                "name": "Anime",
                "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
            },
            {
                "name": "Digital Art",
                "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
            },
            {
                "name": "Photographic",
                "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
            },
            {
                "name": "Pixel art",
                "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
            },
            {
                "name": "Fantasy art",
                "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
            },
            {
                "name": "Neonpunk",
                "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
            },
            {
                "name": "Manga",
                "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
            },
        ]

        self.styles = {k["name"]: k["prompt"] for k in self.style_list}
        self.STYLE_NAMES = list(self.styles.keys())
        self.DEFAULT_STYLE_NAME = "Fantasy art"
        self.MAX_SEED = np.iinfo(np.int32).max

        self.scripts = """
        async () => {
            globalThis.theSketchDownloadFunction = () => {
                console.log("test")
                var link = document.createElement("a");
                dataUri = document.getElementById('download_sketch').href
                link.setAttribute("href", dataUri)
                link.setAttribute("download", "sketch.png")
                document.body.appendChild(link); // Required for Firefox
                link.click();
                document.body.removeChild(link); // Clean up

                // also call the output download function
                theOutputDownloadFunction();
            return false
            }

            globalThis.theOutputDownloadFunction = () => {
                console.log("test output download function")
                var link = document.createElement("a");
                dataUri = document.getElementById('download_output').href
                link.setAttribute("href", dataUri);
                link.setAttribute("download", "output.png");
                document.body.appendChild(link); // Required for Firefox
                link.click();
                document.body.removeChild(link); // Clean up
            return false
            }

            globalThis.UNDO_SKETCH_FUNCTION = () => {
                console.log("undo sketch function")
                var button_undo = document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(1)');
                // Create a new 'click' event
                var event = new MouseEvent('click', {
                    'view': window,
                    'bubbles': true,
                    'cancelable': true
                });
                button_undo.dispatchEvent(event);
            }

            globalThis.DELETE_SKETCH_FUNCTION = () => {
                console.log("delete sketch function")
                var button_del = document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(2)');
                // Create a new 'click' event
                var event = new MouseEvent('click', {
                    'view': window,
                    'bubbles': true,
                    'cancelable': true
                });
                button_del.dispatchEvent(event);
            }

            globalThis.togglePencil = () => {
                el_pencil = document.getElementById('my-toggle-pencil');
                el_pencil.classList.toggle('clicked');
                // simulate a click on the gradio button
                btn_gradio = document.querySelector("#cb-line > label > input");
                var event = new MouseEvent('click', {
                    'view': window,
                    'bubbles': true,
                    'cancelable': true
                });
                btn_gradio.dispatchEvent(event);
                if (el_pencil.classList.contains('clicked')) {
                    document.getElementById('my-toggle-eraser').classList.remove('clicked');
                    document.getElementById('my-div-pencil').style.backgroundColor = "gray";
                    document.getElementById('my-div-eraser').style.backgroundColor = "white";
                }
                else {
                    document.getElementById('my-toggle-eraser').classList.add('clicked');
                    document.getElementById('my-div-pencil').style.backgroundColor = "white";
                    document.getElementById('my-div-eraser').style.backgroundColor = "gray";
                }
            }

            globalThis.toggleEraser = () => {
                element = document.getElementById('my-toggle-eraser');
                element.classList.toggle('clicked');
                // simulate a click on the gradio button
                btn_gradio = document.querySelector("#cb-eraser > label > input");
                var event = new MouseEvent('click', {
                    'view': window,
                    'bubbles': true,
                    'cancelable': true
                });
                btn_gradio.dispatchEvent(event);
                if (element.classList.contains('clicked')) {
                    document.getElementById('my-toggle-pencil').classList.remove('clicked');
                    document.getElementById('my-div-pencil').style.backgroundColor = "white";
                    document.getElementById('my-div-eraser').style.backgroundColor = "gray";
                }
                else {
                    document.getElementById('my-toggle-pencil').classList.add('clicked');
                    document.getElementById('my-div-pencil').style.backgroundColor = "gray";
                    document.getElementById('my-div-eraser').style.backgroundColor = "white";
                }
            }
        }
        """

    def update_canvas(self, use_line, use_eraser):
        if use_eraser:
            _color = "#ffffff"
            brush_size = 20
        if use_line:
            _color = "#000000"
            brush_size = 4
        return gr.update(brush_radius=brush_size, brush_color=_color, interactive=True)

    def upload_sketch(self, file):
        _img = Image.open(file.name)
        _img = _img.convert("L")
        return gr.update(value=_img, source="upload", interactive=True)

    def make_demo(self, fn: Callable):
        """
        역할:
            Gradio UI를 생성하는 헬퍼 함수입니다.
        동작:
            run 함수를 Gradio 앱에 연결합니다.
        """
        with gr.Blocks(css="style.css") as demo:
            with gr.Tab("Sketch"):
                self.run_with_draw(demo, fn)  # 반환값 사용하지 않고 호출만
            with gr.Tab("Image Upload"):
                self.run_with_image(demo, fn)  # 반환값 사용하지 않고 호출만
        return demo

    def run_with_image(self, demo, fn: Callable):
        gr.Markdown("## 이미지 업로드 → AI 이미지 변환")
        with gr.Row():
            with gr.Column():
                upload_image = gr.Image(
                    label="업로드 이미지",
                    type="pil",
                    image_mode="RGB",
                    height=440,
                    width=440,
                    elem_id="upload_input_image",
                    show_label=True,
                )
                prompt = gr.Textbox(label="Prompt", value="", show_label=True)
                style = gr.Dropdown(
                    label="Style",
                    choices=self.STYLE_NAMES,
                    value=self.DEFAULT_STYLE_NAME,
                )
                seed = gr.Textbox(label="Seed", value=42)
                run_button = gr.Button("Run", min_width=50)
            with gr.Column():
                result = gr.Image(
                    label="Result",
                    height=440,
                    width=440,
                    elem_id="upload_output_image",
                    show_label=True,
                    show_download_button=True,
                )

        def upload_run(image, prompt, style, seed):
            # prompt_template은 style에 따라 자동 생성
            prompt_template = self.styles[style]
            # seed는 int로 변환
            try:
                seed = int(seed)
            except Exception:
                seed = 42
            # 이미지가 None이면 빈 이미지로 대체
            if image is None:
                image = Image.new("RGB", (512, 512), (255, 255, 255))
            else:
                image = image.resize((512, 512), Image.LANCZOS)  # 업로드 이미지를 512x512로 리사이즈
            return fn(image, prompt, prompt_template, style, seed)[0]

        run_button.click(
            upload_run,
            inputs=[upload_image, prompt, style, seed],
            outputs=[result],  # gr.update() 제거
        )
        # return 문 제거

    def run_with_draw(self, demo, fn: Callable):
        # these are hidden buttons that are used to trigger the canvas changes
        line = gr.Checkbox(label="line", value=False, elem_id="cb-line")
        eraser = gr.Checkbox(label="eraser", value=False, elem_id="cb-eraser")
        with gr.Row(elem_id="main_row"):
            with gr.Column(elem_id="column_input"):
                gr.Markdown("## INPUT", elem_id="input_header")
                image = gr.Image(
                    source="canvas",
                    tool="color-sketch",
                    type="pil",
                    image_mode="L",
                    invert_colors=True,
                    shape=(512, 512),
                    brush_radius=4,
                    height=440,
                    width=440,
                    brush_color="#000000",
                    interactive=True,
                    show_download_button=True,
                    elem_id="input_image",
                    show_label=False,
                )
                download_sketch = gr.Button("Download sketch", scale=1, elem_id="download_sketch")

                gr.HTML(
                    """
                <div class="button-row">
                    <div id="my-div-pencil" class="pad2"> <button id="my-toggle-pencil" onclick="return togglePencil(this)"></button> </div>
                    <div id="my-div-eraser" class="pad2"> <button id="my-toggle-eraser" onclick="return toggleEraser(this)"></button> </div>
                    <div class="pad2"> <button id="my-button-undo" onclick="return UNDO_SKETCH_FUNCTION(this)"></button> </div>
                    <div class="pad2"> <button id="my-button-clear" onclick="return DELETE_SKETCH_FUNCTION(this)"></button> </div>
                    <div class="pad2"> <button href="TODO" download="image" id="my-button-down" onclick='return theSketchDownloadFunction()'></button> </div>
                </div>
                """
                )

                prompt = gr.Textbox(label="Prompt", value="", show_label=True)
                with gr.Row():
                    style = gr.Dropdown(
                        label="Style",
                        choices=self.STYLE_NAMES,
                        value=self.DEFAULT_STYLE_NAME,
                        scale=1,
                    )
                    prompt_temp = gr.Textbox(
                        label="Prompt Style Template",
                        value=self.styles[self.DEFAULT_STYLE_NAME],
                        scale=2,
                        max_lines=1,
                    )

                with gr.Row():
                    seed = gr.Textbox(label="Seed", value=42, scale=1, min_width=50)
                    randomize_seed = gr.Button("Random", scale=1, min_width=50)

            with gr.Column(elem_id="column_process", min_width=50, scale=1):  # scale=0.4 -> scale=1
                gr.Markdown("## 아가들도 화가로!!", elem_id="description")
                run_button = gr.Button("Run", min_width=50)

            with gr.Column(elem_id="column_output"):
                gr.Markdown("## OUTPUT", elem_id="output_header")
                result = gr.Image(
                    label="Result",
                    height=440,
                    width=440,
                    elem_id="output_image",
                    show_label=False,
                    show_download_button=True,
                )
                download_output = gr.Button("Download output", elem_id="download_output")
                gr.Markdown("### Instructions")
                gr.Markdown("**1**. Enter a text prompt (e.g. cat)")
                gr.Markdown("**2**. Start sketching")
                gr.Markdown("**3**. Change the image style using a style template")
                gr.Markdown("**4**. Try different seeds to generate different results")

        eraser.change(
            fn=lambda x: gr.update(value=not x),
            inputs=[eraser],
            outputs=[line],
            queue=False,
            api_name=False,
        ).then(self.update_canvas, [line, eraser], [image])
        line.change(
            fn=lambda x: gr.update(value=not x),
            inputs=[line],
            outputs=[eraser],
            queue=False,
            api_name=False,
        ).then(self.update_canvas, [line, eraser], [image])

        demo.load(None, None, None, _js=self.scripts)
        randomize_seed.click(
            lambda x: np.random.randint(0, self.MAX_SEED),
            inputs=[],
            outputs=seed,
            queue=False,
            api_name=False,
        )
        inputs = [image, prompt, prompt_temp, style, seed]
        outputs = [result, download_sketch, download_output]
        prompt.submit(fn=fn, inputs=inputs, outputs=outputs, api_name=False)
        style.change(
            lambda x: self.styles[x],
            inputs=[style],
            outputs=[prompt_temp],
            queue=False,
            api_name=False,
        ).then(
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            api_name=False,
        )
        run_button.click(fn=fn, inputs=inputs, outputs=outputs, api_name=False)
        image.change(fn, inputs=inputs, outputs=outputs, queue=False, api_name=False)
