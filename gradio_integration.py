import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import shutil
from datetime import datetime
from uuid import uuid4

from PIL import Image
import numpy as np
import torch
import gradio as gr
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoPipelineForImage2Image
from diffusers import DDPMScheduler, LCMScheduler
import trimesh

from gradio_configs import parse_config
from pipeline.prompt import azim_text_prompt, azim_neg_text_prompt
from ip_adapter import IPAdapterXL
from pipeline.pipeline_controlnet_sd_xl import StableStyleMVDPipeline, get_conditioning_images, get_canny_conditioning_images
from canvas import load_models, cache_path
from diffusers.utils import load_image
from os import path

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

canvas_size = 512

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"


def glb2obj(glb_path, obj_path):
    print('Converting glb to obj')
    mesh = trimesh.load(glb_path)
    
    if isinstance(mesh, trimesh.Scene):
        vertices = 0
        for g in mesh.geometry.values():
            vertices += g.vertices.shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices.shape[0]
    else:
        raise ValueError('It is not mesh or scene')
    
    if vertices > 300000:
        raise ValueError('Too many vertices')
    if not os.path.exists(os.path.dirname(obj_path)):
        os.makedirs(os.path.dirname(obj_path))
    mesh.export(obj_path)
    print('Convert Done')

def resize(image:Image, ratio:float):
    height, width = int(image.height * ratio), int(image.width * ratio)
    
    resized = image.resize((width, height))
    return resized

def center_crop(image, size=512):
    if isinstance(image, Image.Image):
        is_pil_image = True
        image = np.array(image)
    else:
        is_pil_image = False

    h, w = image.shape[:2]
    if h < size or w < size:
        raise ValueError('Image size must be larger than crop size')
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    x2 = x1 + size
    y2 = y1 + size

    if is_pil_image:
        image = Image.fromarray(image[y1:y2, x1:x2])
    else:
        image = image[y1:y2, x1:x2]
    return image


def get_examples():
    example_case = [
        [
            './data_style/lol/3.png',
            './data_mesh/face/face.glb'
        ],
        [
            './data_style/ditoland/7.png',
            './data_mesh/MCM_bag/MCM_Bag_Fix.glb'
        ],
        [
            './data_style/battleground/4.png',
            './data_mesh/sneaker/sneaker.glb'
        ]
    ]
    return example_case


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def style_transfer(style_image:Image, mesh_path:str, cond_type, control_scale,
                   n_prompt, neg_content_prompt, neg_content_scale, 
                   guidance_scale, num_samples, num_inference_steps, seed):
    opt = parse_config()
    
    # make a unique directory for the results
    uuid = str(uuid4())
    result_dir = os.path.join('gradio_results', uuid)
    os.makedirs(result_dir)
    
    if opt.mesh_config_relative:
        # if zip file, extract it and get the obj file path
        if os.path.splitext(mesh_path)[1] == '.glb':
            glb2obj(mesh_path, os.path.join(result_dir, 'export.obj'))
            mesh_path = os.path.join(result_dir, 'export.obj')
        else:
            raise gr.Error('Only glb file is supported')
    
    output_name_components = []
    if opt.prefix and opt.prefix != "":
        output_name_components.append(opt.prefix)
    output_name = "_".join(output_name_components)
    output_dir = os.path.join(result_dir, output_name)
    
    os.makedirs(output_dir)
    print(f"Saving to {output_dir}")
        
    logging_config = {
        "output_dir":output_dir, 
        }
    
    # Model Loading ---------------------------------------------------------------
    print('Model Loading')
    # Load the controlnet model
    if opt.cond_type == "canny":
        controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
    elif opt.cond_type == "depth":
        controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
    controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)


    # load SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)


    # load StyleMVD pipeline
    mvd_pipe = StableStyleMVDPipeline(**pipe.components)
    mvd_pipe.set_config(
        mesh_path=mesh_path,
        mesh_transform={"scale":opt.mesh_scale},
        mesh_autouv=opt.mesh_autouv,
        camera_azims=opt.camera_azims,
        top_cameras=not opt.no_top_cameras,
        texture_size=opt.latent_tex_size,
        render_rgb_size=opt.rgb_view_size,
        texture_rgb_size=opt.rgb_tex_size,
        height=opt.latent_view_size*8,
        width=opt.latent_view_size*8,
        max_batch_size=48,
        controlnet_conditioning_end_scale= opt.conditioning_scale_end,
        guidance_rescale = opt.guidance_rescale,
        multiview_diffusion_end=opt.mvd_end,
        shuffle_background_change=opt.shuffle_bg_change,
        shuffle_background_end=opt.shuffle_bg_end,
        ref_attention_end=opt.ref_attention_end,
        logging_config=logging_config,
        cond_type=opt.cond_type,
    )
    mvd_pipe.initialize_pipeline(
        mesh_path=mesh_path,
        mesh_transform={"scale":opt.mesh_scale},
        mesh_autouv=opt.mesh_autouv,
        camera_azims=opt.camera_azims,
        camera_centers=None,
        top_cameras=not opt.no_top_cameras,
        ref_views=[],
        latent_size=mvd_pipe.height//8,
        render_rgb_size=mvd_pipe.render_rgb_size,
        texture_size=mvd_pipe.texture_size,
        texture_rgb_size=mvd_pipe.texture_rgb_size,
        max_batch_size=mvd_pipe.max_batch_size,
        logging_config=logging_config
    )
    
    ip_model = IPAdapterXL(mvd_pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])
    print('Model Loaded')
    # -----------------------------------------------------------------------------
        
    # generate prompt based on camera pose 
    prompt_list = [azim_text_prompt(opt.prompt, pose) for pose in mvd_pipe.camera_poses]
    negative_prompt_list = [azim_neg_text_prompt(n_prompt, pose) for pose in mvd_pipe.camera_poses]

    # resize and crop to (512, 512)
    width, height = style_image.size
    if min(height, width) < 512:
        raise gr.Error('Style image should be at least 512x512')
    
    ratio = 512 / min(height, width)
    style_image = resize(style_image, ratio)
    style_image = center_crop(style_image, size=512)

    # control image
    if opt.cond_type == "depth":
        conditioning_images, _ = get_conditioning_images(mvd_pipe.uvp, mvd_pipe.height, cond_type=mvd_pipe.cond_type)
    elif opt.cond_type == "canny":
        conditioning_images, _ = get_canny_conditioning_images(mvd_pipe.uvp, mvd_pipe.height)
    

    # generate image
    print('Generating Image')
    _ = ip_model.generate(pil_image=style_image,
                                prompt=prompt_list,
                                negative_prompt=negative_prompt_list,
                                scale=opt.ip_adapter_scale,
                                guidance_scale=guidance_scale,
                                num_samples=num_samples,
                                num_inference_steps=num_inference_steps,
                                seed=seed,
                                image=conditioning_images,
                                controlnet_conditioning_scale=control_scale,
                                )

    input_mesh_render_views = os.path.join(result_dir, 'init_mesh',  'init_mesh.jpg')
    output_mesh_render_views = os.path.join(output_dir, 'results',  'textured_views_rgb.jpg')
    
    stylized_mesh = os.path.join(output_dir, 'results',  'textured.glb')
    
    del controlnet, pipe, mvd_pipe, ip_model
    torch.cuda.empty_cache()
    
    return stylized_mesh, input_mesh_render_views, output_mesh_render_views

# Description
title = r"""
<h1 align="center">StyleMVD</h1>
"""

description = r"""
How to use:<br>
1. Upload a style image.
2. Upload a mesh file.
3. Set stylization mode, only use style block by default.
4. Click the <b>Submit</b> button to begin customization.
5. Share your stylized photo with your friends and enjoy! 😊

Advanced usage:<br>
1. Click advanced options.
2. Upload another source image for image-based stylization using ControlNet.
3. Enter negative content prompt to avoid content leakage.
"""

if __name__ == "__main__":
    if not path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
    
    block = gr.Blocks().queue(max_size=1, api_open=False)
    with block as demo:
        infer = load_models()
        # description
        gr.Markdown(title)
        gr.Markdown(description)
        
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    s = gr.Slider(label="steps", minimum=4, maximum=8, step=1, value=4, interactive=True)
                    c = gr.Slider(label="cfg", minimum=0.1, maximum=3, step=0.1, value=1, interactive=True)
                    i_s = gr.Slider(label="sketch strength", minimum=0.1, maximum=0.9, step=0.1, value=0.9, interactive=True)
                with gr.Column():
                    mod = gr.Text(label="Model Hugging Face id (after changing this wait until the model downloads in the console)", value="Lykon/dreamshaper-8-lcm", interactive=True)
                    t = gr.Text(label="Prompt", value="Scary warewolf, 8K, realistic, colorful, long sharp teeth, splash art", interactive=True)
                    se = gr.Number(label="seed", value=1337, interactive=True)
        with gr.Row(equal_height=True):
            # left column
            i = gr.Paint(canvas_size=(canvas_size, canvas_size), width=canvas_size, height=canvas_size, image_mode="RGB", interactive=True)
            o = gr.Image(width=canvas_size, height=canvas_size, interactive=True)
            
            def process_image(p, im, steps, cfg, image_strength, seed):
                if not im:
                    return Image.new("RGB", (canvas_size, canvas_size))
                if isinstance(im, dict):
                    im = np.array(im['composite'], dtype=np.uint8)
                    im = Image.fromarray(im)
                elif not isinstance(im, Image.Image):
                    im = Image.fromarray(im)
                return infer(
                    prompt=p,
                    image=im,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    strength=image_strength,
                    seed=int(seed)
                )
            
            reactive_controls = [t, i, s, c, i_s, se]
            
            for control in reactive_controls:
                control.change(fn=process_image, inputs=reactive_controls, outputs=o)

            def update_model(model_name):
                global infer
                infer = load_models(model_name)

            mod.change(fn=update_model, inputs=mod)
            
            # middle column
            #with gr.Column():
                # style image
                #style_image = gr.Image(label="Style Image", interactive=True, type='pil')
            
            generator = torch.Generator()
            model_id="Lykon/dreamshaper-8-lcm"
            i2i_pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id,
                cache_dir=cache_path,
                safety_checker=None
            )
            i2i_pipe.scheduler = LCMScheduler.from_config(i2i_pipe.scheduler.config)
            i2i_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            i2i_pipe.fuse_lora()
            device = "cuda" if torch.cuda.is_available() else "mps"
            i2i_pipe.to(device=device)
            
            # style_image to 3d mesh? wondering how to do this
            style_image = o
                        
            # right column
            with gr.Column():
                # mesh file
                mesh_path = gr.Model3D(label="input glb file", interactive=True)
            
        cond_type = gr.Radio(['depth', 'canny'], 
                                value="depth",
                                label="control type",
                                interactive=True)
        with gr.Column():
            # Advanced options
            with gr.Accordion(open=False, label="Advanced Options"):
                control_scale = gr.Slider(minimum=0,maximum=1.0, step=0.01,value=0.5, label="Controlnet conditioning scale", interactive=True)
                n_prompt = gr.Textbox(label="Neg Prompt", value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry", interactive=True)
                
                neg_content_prompt = gr.Textbox(label="Neg Content Prompt", value="", interactive=True)
                neg_content_scale = gr.Slider(minimum=0, maximum=1.0, step=0.01,value=0.5, label="Neg Content Scale", interactive=True)

                guidance_scale = gr.Slider(minimum=1,maximum=15.0, step=0.01,value=5.0, label="guidance scale", interactive=True)
                num_samples= gr.Slider(minimum=1,maximum=4.0, step=1.0,value=1.0, label="num samples", interactive=False)
                num_inference_steps = gr.Slider(minimum=5,maximum=50.0, step=1.0,value=20, label="num inference steps", interactive=True)
                seed = gr.Slider(minimum=-1000000,maximum=1000000,value=1, step=1, label="Seed Value", interactive=True)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
        
        # submit button
        submit = gr.Button("Transfer mesh style", interactive=True)
        with gr.Row(variant='panel'):
            with gr.Column():
                gr.Examples(
                    examples=get_examples(),
                    inputs = [style_image, mesh_path]
                )
            with gr.Column():
                result_model = gr.Model3D(label="Stylized mesh", interactive=False)
                    
        with gr.Column():
            # input mesh render views
            input_mesh_render_views = gr.Image(label="input mesh render views", interactive=False)
            
            # output mesh render views
            output_mesh_render_views = gr.Image(label="stylized mesh render views", interactive=False)
        
        submit.click(
            fn = randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False
        ).then(
            fn=style_transfer,
            inputs=[style_image, mesh_path, cond_type, control_scale,
                    n_prompt, neg_content_prompt, neg_content_scale, 
                    guidance_scale, num_samples, num_inference_steps, 
                    seed],
            outputs=[result_model, input_mesh_render_views, output_mesh_render_views]
        )
    
    demo.launch(share=True)