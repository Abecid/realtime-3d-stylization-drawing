import os
import argparse
import gradio as gr
from canvas import load_models, cache_path
from PIL import Image
from os import path
import numpy as np

# 상수 정의
CANVAS_SIZE = 512
ASSETS_DIR = "assets"
PAINT_IMAGE_PATH = os.path.join(ASSETS_DIR, "paint.png")

# 필요한 디렉토리 생성
os.makedirs(cache_path, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# 초기 모델 로드
infer = load_models()

def process_image(prompt, sketch, steps, cfg, sketch_strength, seed):
    """
    입력된 스케치와 파라미터를 바탕으로 이미지를 생성하고,
    assets/paint.png에 실시간으로 저장합니다.
    """
    if not sketch:
        output_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE))
    else:
        if isinstance(sketch, dict):
            sketch_array = np.array(sketch['composite'], dtype=np.uint8)
            sketch_image = Image.fromarray(sketch_array)
        elif isinstance(sketch, Image.Image):
            sketch_image = sketch
        else:
            sketch_image = Image.fromarray(sketch)
        
        # 모델을 사용하여 이미지 생성
        output_image = infer(
            prompt=prompt,
            image=sketch_image,
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=sketch_strength,
            seed=int(seed)
        )
    
    # 생성된 이미지를 assets/paint.png에 저장
    output_image.save(PAINT_IMAGE_PATH)
    return output_image

def update_model(model_name):
    """
    모델 이름을 업데이트하고 새로운 모델을 로드합니다.
    """
    global infer
    infer = load_models(model_name)
    return f"모델이 {model_name}으로 업데이트 되었습니다."

def load_mesh(mesh_file):
    """
    업로드된 3D 메쉬 파일을 반환합니다.
    """
    if mesh_file is None:
        return None
    # Gradio의 gr.Model3D는 파일 경로 또는 URL을 필요로 합니다.
    # 업로드된 파일의 임시 경로를 반환
    return mesh_file.name

def combine_mesh_paint(mesh_file, paint_image):
    """
    메쉬와 페인트 이미지를 결합하여 새로운 3D 모델을 생성합니다.
    (예시: 페인트 이미지를 메쉬의 텍스처로 적용)
    """
    if mesh_file is None and paint_image is None:
        return None
    
    # 여기에 메쉬와 페인트를 결합하는 로직을 구현합니다.
    # 예시로, 페인트 이미지를 텍스처로 사용하는 경우
    # 실제 구현은 사용하는 3D 라이브러리에 따라 다를 수 있습니다.
    # 현재는 단순히 메쉬 파일 경로를 반환합니다.
    return mesh_file.name if mesh_file else PAINT_IMAGE_PATH

with gr.Blocks() as demo:
    gr.Markdown("## AI 데모: 2D 페인팅 및 3D 모델 뷰어")

    # 모델 상태 표시
    with gr.Row():
        model_status = gr.Textbox(
            label="모델 상태",
            value="로드된 모델: Lykon/dreamshaper-8-lcm",
            interactive=False
        )

    # 컨트롤 패널
    with gr.Column():
        with gr.Row():
            with gr.Column():
                steps_slider = gr.Slider(
                    label="스텝 수",
                    minimum=4,
                    maximum=8,
                    step=1,
                    value=4,
                    interactive=True
                )
                cfg_slider = gr.Slider(
                    label="CFG Scale",
                    minimum=0.1,
                    maximum=3,
                    step=0.1,
                    value=1,
                    interactive=True
                )
                sketch_strength_slider = gr.Slider(
                    label="스케치 강도",
                    minimum=0.1,
                    maximum=0.9,
                    step=0.1,
                    value=0.9,
                    interactive=True
                )
            with gr.Column():
                model_input = gr.Textbox(
                    label="Hugging Face 모델 ID",
                    value="Lykon/dreamshaper-8-lcm",
                    interactive=True
                )
                prompt_input = gr.Textbox(
                    label="프롬프트",
                    value="Scary werewolf, 8K, realistic, colorful, long sharp teeth, splash art",
                    interactive=True
                )
                seed_input = gr.Number(
                    label="시드",
                    value=1337,
                    interactive=True
                )

        with gr.Row(equal_height=True):
            sketch_paint = gr.Paint(
                label="스케치",
                canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
                width=CANVAS_SIZE,
                height=CANVAS_SIZE,
                image_mode="RGB",
                interactive=True
            )
            generated_image = gr.Image(
                label="생성된 이미지",
                width=CANVAS_SIZE,
                height=CANVAS_SIZE,
                interactive=True
            )
        
        # 2D 이미지 생성 및 저장
        reactive_controls_2d = [
            prompt_input,
            sketch_paint,
            steps_slider,
            cfg_slider,
            sketch_strength_slider,
            seed_input
        ]
        for control in reactive_controls_2d:
            control.change(
                fn=process_image,
                inputs=reactive_controls_2d,
                outputs=generated_image
            )
        
        # 모델 업데이트 기능
        model_input.change(
            fn=update_model,
            inputs=model_input,
            outputs=model_status
        )
    
     # 3D 모델 뷰어 섹션
    gr.Markdown("### 3D 모델 뷰어")
    with gr.Row():
        # 첫 번째 3D 모델 뷰어
        with gr.Column():
            mesh_display = gr.Model3D(
                label="3D 모델",
                clear_color=(0.0, 0.0, 0.0, 0.0),
            )
            mesh_input = gr.File(
                label="3D 모델 업로드",
                file_types=["obj", "glb", "gltf", "stl", "ply", "splat"]
            )
            gr.Examples(
                examples=[
                    os.path.join(ASSETS_DIR, "Bunny.obj"),
                    os.path.join(ASSETS_DIR, "Duck.glb"),
                    os.path.join(ASSETS_DIR, "Fox.gltf"),
                    os.path.join(ASSETS_DIR, "face.obj"),
                    os.path.join(ASSETS_DIR, "sofia.stl"),
                    "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/bonsai-7k-mini.splat",
                    "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply",
                ],
                inputs=mesh_input,
                outputs=mesh_display,
                fn=load_mesh,
                label="3D 모델 예제 로드",
                cache_examples=True
            )
        
        # 두 번째 3D 모델 뷰어: Mesh와 Paint를 동시에 처리
        with gr.Column():
            mesh_display_2 = gr.Model3D(
                label="3D 모델 (Mesh + Paint)",
                clear_color=(0.0, 0.0, 0.0, 0.0),
                display_mode="wireframe"
            )
            mesh_input_2 = gr.File(
                label="3D 모델 업로드 (Mesh)",
                file_types=["obj", "glb", "gltf", "stl", "ply", "splat"]
            )
            paint_input_2 = gr.Image(
                label="페인트 이미지 업로드",
                type="filepath"  # 텍스처 적용을 위해 파일 경로로 받음
            )
            combine_button = gr.Button("Mesh와 Paint 결합")
            combine_button.click(
                fn=combine_mesh_paint,
                inputs=[mesh_input_2, paint_input_2],
                outputs=mesh_display_2
            )
            gr.Examples(
                examples=[
                    [os.path.join(ASSETS_DIR, "Bunny.obj"), PAINT_IMAGE_PATH],
                    [os.path.join(ASSETS_DIR, "Duck.glb"), PAINT_IMAGE_PATH],
                    [os.path.join(ASSETS_DIR, "Fox.gltf"), PAINT_IMAGE_PATH],
                    [os.path.join(ASSETS_DIR, "face.obj"), PAINT_IMAGE_PATH],
                    [os.path.join(ASSETS_DIR, "sofia.stl"), PAINT_IMAGE_PATH],
                    ["https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/bonsai-7k-mini.splat", PAINT_IMAGE_PATH],
                    ["https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply", PAINT_IMAGE_PATH],
                ],
                inputs=[mesh_input_2, paint_input_2],
                outputs=mesh_display_2,
                fn=combine_mesh_paint,
                label="3D 모델 + 페인트 예제 로드",
                cache_examples=True
            )
    
    gr.Markdown("### 생성된 이미지는 `assets/paint.png`에 저장됩니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share",
        action="store_true",
        help="Gradio에서 공유를 위해 배포",
        default=False
    )
    args = parser.parse_args()
    demo.launch(share=args.share)