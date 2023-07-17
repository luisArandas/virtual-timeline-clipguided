

# luis arandas 07-03-2023
# single file implementation of a diffusion render w+ virtual 
# timeline and tts, following guided-diffusion and disco-diffusion

import argparse
from pprint import pprint

def main(args):
    pprint(args.__dict__['render'])

    # setup imports 

    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

    import sys
    import re
    import math
    import gc
    import random
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from types import SimpleNamespace

    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from transformers import GPTJForCausalLM, pipeline
    import cv2
    from PIL import Image
    import numpy as np

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root_path = os.getcwd()

    def create_path(filepath):
        os.makedirs(filepath, exist_ok=True)

    out_dir_path = f'{root_path}/output'
    out_dir_prefix = 'render'
    create_path(out_dir_path)
    model_path = f'{root_path}/models'
    create_path(model_path)
    root_path = os.getcwd()

    # clean previous images from root

    for file in os.listdir('.'):
        if file.endswith('.png'):
            os.remove(file)
    img = Image.new('RGB', (1920, 1080), color='black')
    img.save(f'{root_path}/prevFrame.png')

    try:
        sys.path.append(root_path)
        sys.path.append(root_path)
        sys.path.append(f'{root_path}/libs/ResizeRight')
        sys.path.append(f'{root_path}/libs/MiDaS')
        sys.path.append(f'{root_path}/libs/pytorch3d-lite')
        sys.path.append(f'{root_path}/libs/guided_diffusion')
        sys.path.append(f'{root_path}/libs/AdaBins')
    except:
        pass

    import disco_utils as dxf
    import py3d_tools as p3dT

    print("current root path -> ", root_path)
    print("current model path -> ", model_path)
    print("current device -> ", device)

    # specify output directory and create new folder

    existing_folders = [f for f in os.listdir(out_dir_path) if f.startswith(out_dir_prefix)]
    if len(existing_folders) > 0:
        max_suffix = max([int(f[len(out_dir_prefix):]) for f in existing_folders])
    else:
        max_suffix = 0
    render_folder = out_dir_prefix + str(max_suffix + 1)
    new_folder_path = os.path.join(out_dir_path, render_folder)
    os.mkdir(new_folder_path)

    # render-specific variables

    diffusion_sampling_mode = 'ddim'
    animation_mode = '3D'
    batch_name = render_folder
    steps = int(args.__dict__['render'][0])
    timestep_respacing = f'{diffusion_sampling_mode}{steps}'
    diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
    width_height = [512,288] # (16:9 on 3070, cropped)
    clip_guidance_scale = int(args.__dict__['render'][1])
    tv_scale = 0
    range_scale = 0
    cutn_batches = 4
    skip_augs = False
    init_image = None # 'prevFrame.png'
    init_scale = 500
    skip_steps = int(args.__dict__['render'][2])
    frames_scale = 1500
    frames_skip_steps = f'{args.__dict__["render"][3]}%'

    use_secondary_model = True
    if use_secondary_model == True:
        secondary_model = dxf.SecondaryDiffusionImageNet()
        secondary_model.load_state_dict(torch.load(f'{model_path}/secondary_model_imagenet.pth', map_location='cpu'))
        secondary_model.eval().requires_grad_(False).to(device)
    else:
        secondary_model = None

    # field of view variables

    key_frames = True 
    angle = "0:(0)"
    zoom = "0: (1), 10: (1.05)"
    translation_x = f"0: ({args.__dict__['render'][4]})"
    translation_y = f"0: ({args.__dict__['render'][5]})"
    translation_z = f"0: ({args.__dict__['render'][6]})"
    rotation_3d_x = f"0: ({args.__dict__['render'][7]})" 
    rotation_3d_y = f"0: ({args.__dict__['render'][8]})" 
    rotation_3d_z = f"0: ({args.__dict__['render'][9]})"
    midas_weight = 0.9
    near_plane = 200
    far_plane = 1000
    fov = 60

    TRANSLATION_SCALE = 1.0/200.0

    # formal size crop

    side_x = (width_height[0]//64)*64
    side_y = (width_height[1]//64)*64
    if side_x != width_height[0] or side_y != width_height[1]:
        print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

    clamp_grad = True 
    clamp_max = 0.05
    clip_denoised = False
    rand_mag = 0.05

    # format: `[40]*400+[20]*600` = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000
    # cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the 
    # entire image and are good for early structure, innercuts are your standard cutn.

    cut_overview = "[12]*400+[4]*600"
    cut_innercut ="[4]*400+[12]*600"
    cut_ic_pow = 1
    cut_icgray_p = "[0.2]*400+[0]*600"

    display_rate = 10
    batch_size = 1
    skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
    start_frame = 0
    batchNum = 0

    print(f'starting run: {batch_name}({batchNum}) at frame {start_frame}')
    random.seed()
    seed = random.randint(0, 2**32)
    normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # create batch folder for audios and images

    batch_folder = f'{out_dir_path}/{batch_name}'
    create_path(batch_folder)

    # load models which interoperate with the diffusion procedure
    # open_clip, ddim, tacotron2, hifigan, midas

    # setting up open_clip to get image and text embeddings

    import open_clip # working fp32
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', device=device, pretrained='laion400m_e32')
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

    # setting up midas to get depth maps and cameras

    from midas.dpt_depth import DPTDepthModel
    from midas.transforms import Resize, NormalizeImage, PrepareForNet

    def init_midas_depth_model(optimize=True):

        midas_model = DPTDepthModel(
            path=f"{model_path}/dpt_large-midas-2f21e586.pt",
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        midas_transform = T.Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        midas_model.eval()
        
        if device == torch.device("cuda"):
            midas_model = midas_model.to(memory_format=torch.channels_last)  
            midas_model = midas_model.half()

        midas_model.to(device)

        return midas_model, midas_transform, net_w, net_h, resize_mode, normalization

    midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model("dpt_large")

    # setting up adabins to cross with midas
    sys.path.append(f'{root_path}/libs/AdaBins')
    from infer import InferenceHelper

    # setting up unet model and ddim progressive diffusion

    sys.path.append(f'{root_path}/libs/guided-diffusion')
    # probably wont do gd full import in the future, exposing the steps
    import guided_diffusion.gaussian_diffusion as GaussianDiffusion
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, create_model
    from guided_diffusion.respace import SpacedDiffusion, space_timesteps
    from guided_diffusion.unet import UNetModel

    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': True,
        'use_fp16': True,
        'use_scale_shift_norm': True,
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
    })

    print("model_config", model_config)

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=GaussianDiffusion.get_named_beta_schedule("linear", steps),
        model_mean_type=(
            GaussianDiffusion.ModelMeanType.EPSILON if not model_config['predict_xstart'] else GaussianDiffusion.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                GaussianDiffusion.ModelVarType.FIXED_LARGE # GaussianDiffusion.ModelVarType.FIXED_SMALL
            )
            if not model_config['learn_sigma']
            else GaussianDiffusion.ModelVarType.LEARNED_RANGE
        ),
        loss_type=GaussianDiffusion.LossType.MSE,
        rescale_timesteps=model_config['rescale_timesteps'],
    )

    # setting up diffusion model

    diffusion_model = create_model(
        256, # image size
        256, # num channels
        2, # num res blocks
        channel_mult="",#"1,1,2,2,4,4", # image size 256
        learn_sigma=model_config['learn_sigma'],
        class_cond=model_config['class_cond'],
        use_checkpoint=True,
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=4,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
    )

    # model2, _diffusion = create_model_and_diffusion(**model_config)
    diffusion_model.load_state_dict(torch.load(f'{model_path}/256x256_diffusion_uncond.pt', map_location='cpu'), strict=False)
    # do jit.trace for a quick load without creating the net object and c++
    diffusion_model.requires_grad_(False).eval().to(device)
    diffusion_model.convert_to_fp16()

    for name, param in diffusion_model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()

    # setup tts
    
    # currently commented as it is 12gb from standard download I believe
    # gpt_generator = GPTJForCausalLM.from_pretrained(
    #     "EleutherAI/gpt-j-6B",
    #         revision="float16",
    #         torch_dtype=torch.float16,
    #         low_cpu_mem_usage=True
    # )
    # gen = pipeline("text-generation", model=gpt_generator, tokenizer=tokenizer, device=device)


    from timeline import VirtualTimelineStore

    timeline = VirtualTimelineStore()
    timeline.batch_folder = batch_folder

    # sample text for testing
    with open(os.path.join(root_path + os.sep + "input.txt"), 'r') as fd:
        timeline.starting_text = fd.read()
        print("raw text ->", timeline.starting_text)

    # define internal timeline variables according to first LED(t)

    sum_text, sum_scores = timeline.get_summary(
        timeline.starting_text,
        model_size="small",
        num_beams=2,
        token_batch_length=1024,
        length_penalty=0.3,
        repetition_penalty=3.5,
        no_repeat_ngram_size=3
    )

    # added grammar corrector just in case
    timeline.short_summary = str(sum_text) # timeline.flan_grammar_correct(str(sum_text))
    timeline.short_summary_sentences = [s.strip() for s in re.split('[,.;]', str(sum_text[0])) if s.strip()]

    # timeline.short_summary_sentences = [timeline.flan_grammar_correct(str(sentence)) for sentence in timeline.short_summary_sentences]
    # timeline.keywords = (further inference with gpt-j-6b)

    print("short summary -> ", timeline.short_summary)
    print("short summary sentences -> ", timeline.short_summary_sentences)

    # running three types of sequencing, there is no compensation if summary procedures fail
    # please feel free to comment as you go, will add adaptibility in further commands

    # 1) based on sequence length from user input, space prompt summary outputs with equal distances at defined FPS and compute their TTS separately to the same folder
    # 2) without sequence length definition calculate sequence length from TTS in seconds and compensate one each side
    # 3) without sequence length definition do step 2) but compute TTS separately and establish five second prompts for each summarised sentence

    timeline.fps = 25
    timeline.film_length_secs = 60

    # generate audio files
    timeline.tts_prompts_by_sentence()

    # generate text prompts to further CLIP embeddings
    timeline.sequencer_1()
    # timeline.sequencer_2()
    # timeline.sequencer_3()
    # timeline.sequencer_random()

    max_frames = timeline.max_frames
    text_prompts = timeline.data[1]
    print("timeline keyframes ->", timeline.data)
    print("max frames ->", max_frames)
    torch.cuda.empty_cache()

    # we appropriate the namespace and add it to the timeline object
    # this way we keep track of previous renders to compare

    if key_frames:
        angle_series = dxf.get_inbetweens(dxf.parse_key_frames(angle), max_frames)
        zoom_series = dxf.get_inbetweens(dxf.parse_key_frames(zoom), max_frames)
        translation_x_series = dxf.get_inbetweens(dxf.parse_key_frames(translation_x), max_frames)
        translation_y_series = dxf.get_inbetweens(dxf.parse_key_frames(translation_y), max_frames)
        translation_z_series = dxf.get_inbetweens(dxf.parse_key_frames(translation_z), max_frames)
        rotation_3d_x_series = dxf.get_inbetweens(dxf.parse_key_frames(rotation_3d_x), max_frames)
        rotation_3d_y_series = dxf.get_inbetweens(dxf.parse_key_frames(rotation_3d_y), max_frames)
        rotation_3d_z_series = dxf.get_inbetweens(dxf.parse_key_frames(rotation_3d_z), max_frames)
    else:
        angle = float(angle)
        zoom = float(zoom)
        translation_x = float(translation_x)
        translation_y = float(translation_y)
        translation_z = float(translation_z)
        rotation_3d_x = float(rotation_3d_x)
        rotation_3d_y = float(rotation_3d_y)
        rotation_3d_z = float(rotation_3d_z)


    # model loading upon step is not optimised, proposed as future work
    
    unet_basemodel_name = 'tf_efficientnet_b5_ap'
    unet_adaptive_bins_model = torch.hub.load('rwightman/gen-efficientnet-pytorch', unet_basemodel_name, pretrained=True)

    render_ns = {
        'batchNum': batchNum,
        'prompts_series':dxf.split_prompts(text_prompts, max_frames) if text_prompts else None,
        'seed': seed,
        'display_rate':display_rate,
        'batch_size':batch_size,
        'batch_name': batch_name,
        'steps': steps,
        'diffusion_sampling_mode': diffusion_sampling_mode,
        'width_height': width_height,
        'clip_guidance_scale': clip_guidance_scale,
        'tv_scale': tv_scale,
        'range_scale': range_scale,
        'cutn_batches': cutn_batches,
        'init_image': init_image,
        'init_scale': init_scale,
        'skip_steps': skip_steps,
        'side_x': side_x,
        'side_y': side_y,
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
        'animation_mode': animation_mode,
        'key_frames': key_frames,
        'max_frames': max_frames if animation_mode != "None" else 1,
        'start_frame': start_frame,
        'angle': angle,
        'zoom': zoom,
        'translation_x': translation_x,
        'translation_y': translation_y,
        'translation_z': translation_z,
        'rotation_3d_x': rotation_3d_x,
        'rotation_3d_y': rotation_3d_y,
        'rotation_3d_z': rotation_3d_z,
        'midas_weight': midas_weight,
        'near_plane': near_plane,
        'far_plane': far_plane,
        'fov': fov,
        'angle_series':angle_series,
        'zoom_series':zoom_series,
        'translation_x_series':translation_x_series,
        'translation_y_series':translation_y_series,
        'translation_z_series':translation_z_series,
        'rotation_3d_x_series':rotation_3d_x_series,
        'rotation_3d_y_series':rotation_3d_y_series,
        'rotation_3d_z_series':rotation_3d_z_series,
        'frames_scale': frames_scale,
        'calc_frames_skip_steps': math.floor(steps * skip_step_ratio),
        'skip_step_ratio': skip_step_ratio,
        'text_prompts': text_prompts,
        'cut_overview': eval(cut_overview),
        'cut_innercut': eval(cut_innercut),
        'cut_ic_pow': cut_ic_pow,
        'cut_icgray_p': eval(cut_icgray_p),
        'clamp_grad': clamp_grad,
        'clamp_max': clamp_max,
        'skip_augs': skip_augs,
        'clip_denoised': clip_denoised,
        'rand_mag': rand_mag,
        'adabins_pretrained_path': str(root_path + os.sep + 'models' + os.sep + 'AdaBins_nyu.pt')
    }

    render_ns = SimpleNamespace(**render_ns)

    # update timeline

    def do_3d_step(img_filepath, frame_num, midas_model, midas_transform):
        if render_ns.key_frames:
            translation_x = render_ns.translation_x_series[frame_num]
            translation_y = render_ns.translation_y_series[frame_num]
            translation_z = render_ns.translation_z_series[frame_num]
            rotation_3d_x = render_ns.rotation_3d_x_series[frame_num]
            rotation_3d_y = render_ns.rotation_3d_y_series[frame_num]
            rotation_3d_z = render_ns.rotation_3d_z_series[frame_num]
            print(f'translation_z: {translation_z}', f'rotation_3d_y: {rotation_3d_y}')

        translate_xyz = [-translation_x*TRANSLATION_SCALE, translation_y*TRANSLATION_SCALE, -translation_z*TRANSLATION_SCALE]
        rotate_xyz_degrees = [rotation_3d_x, rotation_3d_y, rotation_3d_z]
        rotate_xyz = [math.radians(rotate_xyz_degrees[0]), math.radians(rotate_xyz_degrees[1]), math.radians(rotate_xyz_degrees[2])]
        rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)    
        next_step_pil = dxf.transform_image_3d(
            img_filepath, 
            unet_adaptive_bins_model,
            midas_model,
            midas_transform, 
            device,
            rot_mat, 
            translate_xyz, 
            render_ns.near_plane, 
            render_ns.far_plane,
            render_ns.fov, 
            padding_mode='border',
            sampling_mode='bicubic', 
            midas_weight=render_ns.midas_weight,
            adabins_pretrained_path=render_ns.adabins_pretrained_path

        )
        return next_step_pil


    gc.collect()
    torch.cuda.empty_cache()


    # start to compute the diffusion run

    def do_run():

        print("save to file here")
        
        seed = render_ns.seed
        
        # discrete loop over sequence duration
        # established previously

        for frame_num in range(render_ns.start_frame, render_ns.max_frames):

            if render_ns.init_image in ['','none', 'None', 'NONE']:
                init_image = None
            else:
                init_image = render_ns.init_image
            init_scale = render_ns.init_scale
            skip_steps = render_ns.skip_steps
                
            # logic to continue after the first frame

            if frame_num > 0:
                seed += 1
                if frame_num == start_frame:
                    img_filepath = batch_folder+f"/{batch_name}({batchNum})_{start_frame-1:05}.png" # img_filepath = 'prevFrame.png'
                    
                else:
                    img_filepath = 'prevFrame.png'

                next_step_pil = do_3d_step(img_filepath, frame_num, midas_model, midas_transform)
                next_step_pil.save('prevFrameScaled.png')

                init_image = 'prevFrameScaled.png'
                init_scale = render_ns.frames_scale
                skip_steps = render_ns.calc_frames_skip_steps 
            else:
                init_image = None
        
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
        
            target_embeds, weights = [], []
            
            if render_ns.prompts_series is not None and frame_num >= len(render_ns.prompts_series):
                frame_prompt = render_ns.prompts_series[-1]
            elif render_ns.prompts_series is not None:
                frame_prompt = render_ns.prompts_series[frame_num]
            else:
                frame_prompt = []
            
            

            print(f'frame {frame_num} prompt: {frame_prompt}')

            model_stats = []

            # openclip model setup
            
            cutn = 16
            model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
            model_stat["clip_model"] = clip_model
            
            # openclip encoding of each prompt in the object
            # define that encoding as the target embedding
            # concatenate all target embeddings into a single tensor
            # create a new tensor with the weights added in each prompt :(int)
            
            for prompt in frame_prompt:
                txt, weight = dxf.parse_prompt(prompt) # do this in dictionary class
                txt = clip_model.encode_text(tokenizer(prompt).to(device)).float()
                model_stat["target_embeds"].append(txt)
                model_stat["weights"].append(weight)

        
            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

            # if this is not the first diffused frame
            # use it to condition next 
            
            init = None
            if init_image is not None:
                init = Image.open(dxf.fetch(init_image)).convert('RGB')
                init = init.resize((render_ns.side_x, render_ns.side_y), Image.Resampling.LANCZOS)
                init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
            
            cur_t = None

            # function to condition the diffusion run and returning gradient calculation

            def cond_fn(x, t, y=None):
                
                with torch.enable_grad():
                    
                    x_is_NaN = False
                    x = x.detach().requires_grad_()
                    n = x.shape[0]

                    
                    if use_secondary_model is True:
                        alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                        sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                        cosine_t = dxf.alpha_sigma_to_t(alpha, sigma)
                        out = secondary_model(x, cosine_t[None].repeat([n])).pred
                        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                        x_in = out * fac + x * (1 - fac)
                        x_in_grad = torch.zeros_like(x_in)
                    else:
                        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                        out = diffusion.p_mean_variance(diffusion_model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                        x_in = out['pred_xstart'] * fac + x * (1 - fac)
                        x_in_grad = torch.zeros_like(x_in)

                    for model_stat in model_stats:
                        
                        # calculate the loss of each cutn_batch spherical distances

                        for i in range(render_ns.cutn_batches):
                            t_int = int(t.item())+1 # errors on last step without +1, need to find source
                            #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                            try:
                                input_resolution=model_stat["clip_model"].visual.input_resolution
                            except:
                                input_resolution=224

                            cuts = dxf.MakeCutoutsDango(input_resolution,
                                    Overview= render_ns.cut_overview[1000-t_int], 
                                    InnerCrop = render_ns.cut_innercut[1000-t_int], IC_Size_Pow=render_ns.cut_ic_pow, IC_Grey_P = render_ns.cut_icgray_p[1000-t_int]
                                    )
                            clip_in = normalize(cuts(x_in.add(1).div(2)))
                            
                            image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                            
                            dists = dxf.spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                            dists = dists.view([render_ns.cut_overview[1000-t_int]+render_ns.cut_innercut[1000-t_int], n, -1])
                            
                            losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                            x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches

                    tv_losses = dxf.tv_loss(x_in)
                    if use_secondary_model is True:
                        range_losses = dxf.range_loss(out)
                    else:
                        range_losses = dxf.range_loss(out['pred_xstart'])
                    #range_losses = range_loss(out['pred_xstart'])
                    sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
                    sat_scale = 1
                    loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
                    
                    x_in_grad += torch.autograd.grad(loss, x_in)[0]
                    if torch.isnan(x_in_grad).any()==False:
                        grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                    else:
                        x_is_NaN = True
                        grad = torch.zeros_like(x)

                if render_ns.clamp_grad and x_is_NaN == False:
                    magnitude = grad.square().mean().sqrt()
                    return grad * magnitude.clamp(max=render_ns.clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
                return grad
        
            sample_fn = diffusion.ddim_sample_loop_progressive
            
            for i in range(1):
                gc.collect()
                torch.cuda.empty_cache()

                cur_t = diffusion.num_timesteps - skip_steps - 1
                total_steps = cur_t

                samples = sample_fn(
                    diffusion_model,
                    (batch_size, 3, render_ns.side_y, render_ns.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=True,
                    eta=0.8,
                )
                
                for j, sample in enumerate(samples):
                    cur_t -= 1
                    if j % render_ns.display_rate == 0 or cur_t == -1:
                        for k, image in enumerate(sample['pred_xstart']):                            
                            #if args.n_batches > 0:
                            save_num = f'{frame_num:04}'
                            filename = f'{render_ns.batch_name}({render_ns.batchNum})_{save_num}.png'
                            
                            image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                            if j % render_ns.display_rate == 0 or cur_t == -1:
                                image.save('progress.png')

                            image.save('prevFrame.png') # just update
                            image.save(f'{batch_folder}/{filename}')


    do_run()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # probably good for separation -> # parser.add_argument("run_type", help="type of run (sequencer_1 to sequencer_3)", choices=["sequencer_1", "sequencer_2", "sequencer_3"])
    parser.add_argument("-re", "--render", help="int(batch_steps) int(batch_cgs) int(skip_steps) int(frame_skip_steps) float(cam_t_x) float(cam_t_y) float(cam_t_z) float(cam_r_x) float(cam_r_y) float(cam_r_z)", action="store", nargs=10, type=str)
    args = parser.parse_args()
    main(args)
    
