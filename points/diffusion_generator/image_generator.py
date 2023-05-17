from typing import List
import glob
import importlib
import time
import math
import os
import random
import re

import diffusers
import numpy as np
import torch
import torchvision

from pipline_diffusers import *
from diffusers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)

import PIL
import library.model_util as model_util
import tools.original_control_net as original_control_net

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from transformers import CLIPModel
from tools.original_control_net import ControlNetInfo
from XTI_hijack import unet_forward_XTI, downblock_forward_XTI, upblock_forward_XTI


# img2imgの前処理、画像の読み込みなど
def load_images(path):
    if os.path.isfile(path):
        paths = [path]
    else:
        paths = (
            glob.glob(os.path.join(path, "*.png"))
            + glob.glob(os.path.join(path, "*.jpg"))
            + glob.glob(os.path.join(path, "*.jpeg"))
            + glob.glob(os.path.join(path, "*.webp"))
        )
        paths.sort()
    images = []
    for p in paths:
        image = Image.open(p)
        if image.mode != "RGB":
            print(f"convert image to RGB from {image.mode}: {p}")
            image = image.convert("RGB")
        images.append(image)
    return images

def resize_images(imgs, size):
    resized = []
    for img in imgs:
        r_img = img.resize(size, Image.Resampling.LANCZOS)
        if hasattr(img, "filename"):  # filename属性がない場合があるらしい
            r_img.filename = img.filename
        resized.append(r_img)
    return resized

def get_prompt_list(prompt, from_file):
    # promptを取得する
    if from_file is not None:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r", encoding="utf-8") as f:
            prompt_list = f.read().splitlines()
            prompt_list = [d for d in prompt_list if len(d.strip()) > 0]
    elif prompt is not None:
        prompt_list = [prompt]
    else:
        prompt_list = []
    return prompt_list


class GenImages():
    def __init__(self):
        # load Stable Diffusion v2.0 model
        self.v2 = False 
        # enable v-parameterization training
        self.v_parameterization = False

        self.prompt = None
        # if specified, load prompts from this file
        self.from_file = None

        # image to inpaint or to generate from
        self.image_path = None
        # mask in inpainting
        self.mask_path = None
        # img2img strength
        self.strength = None
        # number of images per prompt
        self.images_per_prompt = 1

        self.outdir = None
        # sequential output file name
        self.use_original_file_name = False
        # prepend original file name in img2img
        self.sequential_file_name = False

        # sample this often
        self.n_iter = 1
        self.width = 512
        self.height = 512
        self.batch_size = 1
        self.steps = 50
        # ddim,pndm,lms,euler,euler_a,heun,dpm_2,dpm_2_a,dpmsolver,dpmsolver++,dpmsingle,k_lms,k_euler,k_euler_a,k_dpm_2,k_dpm_2_a,
        self.sampler = "ddim"
        # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale
        self.scale = 7.5
        # layer number from bottom to use in CLIP
        self.clip_skip = 1

        # use fp16 / bfloat16 / float32
        self._dtype = torch.float32
        # path to checkpoint of model
        self._ckpt = None
        # path to checkpoint of vae to replace
        self._vae = None

        # directory for caching Tokenizer (for offline training)
        self.tokenizer_cache_dir = None
        # seed, or seed of seeds in multiple generation
        self.seed = -1

        # use same seed for all prompts in iteration if no seed specified
        self.iter_same_seed = False

        # use xformers
        self.xformers = False
        # use xformers by diffusers (Hypernetworks doesn't work) 
        self.diffusers_xformers = False
        
        # set channels last option to model
        self.opt_channels_last = False

        # additional network module to use, lora
        self.network_module = []
        # additional network weights to load
        self.network_weights = []
        # additional network multiplier
        self.network_mul = []
        # additional argmuments for network (key=value)
        self.network_args = None
        # show metadata of network model
        self.network_show_meta = False
        # "merge network weights to original model
        self.network_merge = False

        # Embeddings files of Extended Textual Inversion / Extended Textual Inversionのembeddings
        self.XTI_embeddings = None
        # Embeddings files of Textual Inversion / Textual Inversionのembeddings
        self.textual_inversion_embeddings = None

        # enable highres fix, reso scale for 1st stage
        self.highres_fix_scale = 0.0
        # 1st stage steps for highres fix
        self.highres_fix_steps = 28
        # upscaler module for highres fix 
        self.highres_fix_upscaler = None
        # additional argmuments for upscaler (key=value)
        self.highres_fix_upscaler_args = None
        # save 1st stage images for highres fix
        self.highres_fix_save_1st = None
        # use latents upscaling for highres fix
        self.highres_fix_latents_upscaling = None

        # set another guidance scale for negative prompt
        self.negative_scale = None

        # enable CLIP guided SD, scale for guidance (DDIM, PNDM, LMS samplers only)
        self.clip_guidance_scale = 0.0
        # enable CLIP guided SD by image, scale for guidance
        self.clip_image_guidance_scale = 0.0
        # image to CLIP guidance 
        self.guide_image_path = None

        self.vgg16_model = None
        # enable VGG16 guided SD by image, scale for guidance
        self.vgg16_guidance_scale = 0.0
        # layer of VGG16 to calculate contents guide (1~30, 20 for conv4_2)
        self.vgg16_guidance_layer = 20
        
        self.max_token_length = 0
        # max embeding multiples, max token length is 75 * multiples
        self.max_embeddings_multiples = 0

        # ControlNet models to use 
        self.control_net_models = None
        # ControlNet preprocess to use
        self.control_net_preps = None
        # ControlNet weights 
        self.control_net_weights = None
        # ControlNet guidance ratio for steps
        self.control_net_ratios = 0.0

        # batch size for VAE, < 1.0 for ratio
        self.vae_batch_size = 0

        # 
        self._networks = []
        self._text_encoder = None
        
        self._unet = None
        self._tokenizer = None
        self._clip_model = None
        self._scheduler_num_noises_per_step = 0
        
        self._noise_manager = None
        self._scheduler = None
        self._upscaler = None
        self._device = None
        self._pipe = None
        self._control_nets: List[ControlNetInfo] = []
        self._network_default_muls = []

        self._init_images = None
        self._mask_images = None
        self._guide_images = None

        self.img_name_prefix = None # "network"
        self.img_name_type = "default"

    def set_dtype(self, dtype:str=["fp16" ,"bp16", "fp32"]):
        if dtype == "fp16":
            self._dtype = torch.float16
        elif dtype == "bp16":
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float32

    def set_ckpt(self, ckpt_path:str):
        self._ckpt = ckpt_path
        # モデルを読み込む
        if not os.path.isfile(ckpt_path):  # ファイルがないならパターンで探し、一つだけ該当すればそれを使う
            files = glob.glob(ckpt_path)
            if len(files) == 1:
                self._ckpt = files[0]
        
        use_stable_diffusion_format = os.path.isfile(self._ckpt)
        if use_stable_diffusion_format:
            print("load StableDiffusion checkpoint")
            self._text_encoder, self._vae, self._unet = model_util.load_models_from_stable_diffusion_checkpoint(self.v2, self._ckpt)
        else:
            print("load Diffusers pretrained models")
            loading_pipe = StableDiffusionPipeline.from_pretrained(self._ckpt, safety_checker=None, torch_dtype=self._dtype)
            self._text_encoder = loading_pipe.text_encoder
            self._vae = loading_pipe.vae
            self._unet = loading_pipe.unet
            self._tokenizer = loading_pipe.tokenizer
            del loading_pipe
        
        if self.clip_guidance_scale  > 0 or self.clip_image_guidance_scale > 0:
            print("prepare clip model")
            self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, torch_dtype=self._dtype)

        if not self.vgg16_guidance_scale:
            print("prepare resnet model")
            self.vgg16_model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.IMAGENET1K_V1)

    def set_vae(self, vae_id):
        self._vae = model_util.load_vae(vae_id, self._dtype)
        print("additional VAE loaded")


    def _load_tokenizer(self):
        print("loading tokenizer")
        if not os.path.isfile(self._ckpt):
            return
        print("prepare tokenizer")
        original_path = V2_STABLE_DIFFUSION_PATH if self.v2 else TOKENIZER_PATH

        tokenizer: CLIPTokenizer = None
        if self.tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(self.tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                print(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)  # same for v1 and v2

        if tokenizer is None:
            if self.v2:
                tokenizer = CLIPTokenizer.from_pretrained(original_path, subfolder="tokenizer")
            else:
                tokenizer = CLIPTokenizer.from_pretrained(original_path)

        if self.max_token_length > 0:
            print(f"update token length: {self.max_token_length}")

        if self.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            print(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        self._tokenizer = tokenizer

    def _load_scheduler(self):
        sched_init_args = {}
        self._scheduler_num_noises_per_step = 1
        if self.sampler == "ddim":
            scheduler_cls = DDIMScheduler
            scheduler_module = diffusers.schedulers.scheduling_ddim
        elif self.sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
            scheduler_cls = DDPMScheduler
            scheduler_module = diffusers.schedulers.scheduling_ddpm
        elif self.sampler == "pndm":
            scheduler_cls = PNDMScheduler
            scheduler_module = diffusers.schedulers.scheduling_pndm
        elif self.sampler == "lms" or self.sampler == "k_lms":
            scheduler_cls = LMSDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_lms_discrete
        elif self.sampler == "euler" or self.sampler == "k_euler":
            scheduler_cls = EulerDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_euler_discrete
        elif self.sampler == "euler_a" or self.sampler == "k_euler_a":
            scheduler_cls = EulerAncestralDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
        elif self.sampler == "dpmsolver" or self.sampler == "dpmsolver++":
            scheduler_cls = DPMSolverMultistepScheduler
            sched_init_args["algorithm_type"] = self.sampler
            scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep
        elif self.sampler == "dpmsingle":
            scheduler_cls = DPMSolverSinglestepScheduler
            scheduler_module = diffusers.schedulers.scheduling_dpmsolver_singlestep
        elif self.sampler == "heun":
            scheduler_cls = HeunDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_heun_discrete
        elif self.sampler == "dpm_2" or self.sampler == "k_dpm_2":
            scheduler_cls = KDPM2DiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_discrete
        elif self.sampler == "dpm_2_a" or self.sampler == "k_dpm_2_a":
            scheduler_cls = KDPM2AncestralDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete
            self._scheduler_num_noises_per_step = 2

        if self.v_parameterization:
            sched_init_args["prediction_type"] = "v_prediction"

        self._noise_manager = NoiseManager()
        if scheduler_module is not None:
            scheduler_module.torch = TorchRandReplacer(self._noise_manager)

        self._scheduler = scheduler_cls(
            num_train_timesteps=SCHEDULER_TIMESTEPS,
            beta_start=SCHEDULER_LINEAR_START,
            beta_end=SCHEDULER_LINEAR_END,
            beta_schedule=SCHEDLER_SCHEDULE,
            **sched_init_args,
        )

        if hasattr(self._scheduler.config, "clip_sample") and self._scheduler.config.clip_sample is False:
            print("set clip_sample to True")
            self._scheduler.config.clip_sample = True

    def _set_highres(self):
         # upscalerの指定があれば取得する
        if self.highres_fix_upscaler:
            print("import upscaler module:", self.highres_fix_upscaler)
            imported_module = importlib.import_module(self.highres_fix_upscaler)

            us_kwargs = {}
            if self.highres_fix_upscaler_args:
                for net_arg in self.highres_fix_upscaler_args.split(";"):
                    key, value = net_arg.split("=")
                    us_kwargs[key] = value

            print("create upscaler")
            self._upscaler = imported_module.create_upscaler(**us_kwargs)
            self._upscaler.to(self._dtype).to(self._device)

    def _set_control_net(self):
        if self.control_net_models:
            for i, model in enumerate(self.control_net_models):
                prep_type = None if not self.control_net_preps or len(self.control_net_preps) <= i else self.control_net_preps[i]
                weight = 1.0 if not self.control_net_weights or len(self.control_net_weights) <= i else self.control_net_weights[i]
                ratio = 1.0 if not self.control_net_ratios or len(self.control_net_ratios) <= i else self.control_net_ratios[i]

                ctrl_unet, ctrl_net = original_control_net.load_control_net(self.v2, self._unet, model)
                prep = original_control_net.load_preprocess(prep_type)
                self._control_nets.append(ControlNetInfo(ctrl_unet, ctrl_net, prep, weight, ratio))
    
    def create_pipline(self):
        # V2配置检查
        assert not self.highres_fix_scale or self.image_path is None, f"highres_fix doesn't work with img2img / highres_fixはimg2imgと同時に使えません"

        if self.v_parameterization and not self.v2:
            print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
        if self.v2 and self.clip_skip > 1:
            print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

        # xformers、Hypernetwork 一致
        if not self.diffusers_xformers:
            replace_unet_modules(self._unet, not self.xformers, self.xformers)

        # load tokenizer token分词器
        self._load_tokenizer()

        # load scheduler 扩散调度器
        self._load_scheduler()

        # 确定 设备类型，使用GPU or CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps"を考量してない

        # custom pipeline
        self._vae.to(self._dtype).to(self._device)
        self._text_encoder.to(self._dtype).to(self._device)
        self._unet.to(self._dtype).to(self._device)
        if self._clip_model is not None:
            self._clip_model.to(self._dtype).to(self._device)
        if self.vgg16_model is not None:
            self.vgg16_model.to(self._dtype).to(self._device)
        
        # set highres
        self._set_highres()
        
        # set ControlNet
        self._set_control_net()
        
        self._pipe = PipelineLike(
            self._device,
            self._vae,
            self._text_encoder,
            self._tokenizer,
            self._unet,
            self._scheduler,
            self.clip_skip,
            self._clip_model,
            self.clip_guidance_scale,
            self.clip_image_guidance_scale,
            self.vgg16_model,
            self.vgg16_guidance_scale,
            self.vgg16_guidance_layer,
        )
        self._pipe.set_control_nets(self._control_nets)
        print("pipeline is ready.")

        if self.diffusers_xformers:
            self._pipe.enable_xformers_memory_efficient_attention()

    def load_network(self, append_network=True):
        # networkを組み込む
        network_list = []
        if len(self.network_module) > 0:
            for i, network_module in enumerate(self.network_module):
                print("import network module:", network_module)
                imported_module = importlib.import_module(network_module)

                network_mul = 1.0 if self.network_mul is None or len(self.network_mul) <= i else self.network_mul[i]
                self._network_default_muls.append(network_mul)

                net_kwargs = {}
                if self.network_args and i < len(self.network_args):
                    network_args = self.network_args[i]
                    # TODO escape special chars
                    network_args = network_args.split(";")
                    for net_arg in network_args:
                        key, value = net_arg.split("=")
                        net_kwargs[key] = value

                if self.network_weights and i < len(self.network_weights):
                    network_weight = self.network_weights[i]
                    print("load network weights from:", network_weight)

                    if model_util.is_safetensors(network_weight) and self.network_show_meta:
                        from safetensors.torch import safe_open

                        with safe_open(network_weight, framework="pt") as f:
                            metadata = f.metadata()
                        if metadata is not None:
                            print(f"metadata for: {network_weight}: {metadata}")
                    network_list.append("%s_%.2f" % (os.path.basename(self.network_weights[0]).split(".")[0], network_mul))
                    network, weights_sd = imported_module.create_network_from_weights(
                        network_mul, network_weight, self._vae, self._text_encoder, self._unet, for_inference=True, **net_kwargs
                    )
                else:
                    raise ValueError("No weight. Weight is required.")
                if network is None:
                    return

                mergiable = hasattr(network, "merge_to")
                if self.network_merge and not mergiable:
                    print("network is not mergiable. ignore merge option.")

                if not self.network_merge or not mergiable:
                    network.apply_to(self._text_encoder, self._unet)
                    info = network.load_state_dict(weights_sd, False)  # network.load_weightsを使うようにするとよい
                    print(f"weights are loaded: {info}")

                    if self.opt_channels_last:
                        network.to(memory_format=torch.channels_last)
                    network.to(self._dtype).to(self._device)

                    if append_network:
                        self._networks.append(network)
                    else:
                        self._networks=[network]
                else:
                    network.merge_to(self._text_encoder, self._unet, weights_sd, self._dtype, self._device)
            if self.img_name_type == "network":
                self.img_name_prefix = ",".join(network_list)
        
        if self.opt_channels_last:
            print(f"set optimizing: channels last")
            self._text_encoder.to(memory_format=torch.channels_last)
            self._vae.to(memory_format=torch.channels_last)
            self._unet.to(memory_format=torch.channels_last)
            if self._clip_model is not None:
                self._clip_model.to(memory_format=torch.channels_last)
            for network in self._networks:
                network.to(memory_format=torch.channels_last)
            if self.vgg16_model is not None:
                self.vgg16_model.to(memory_format=torch.channels_last)

            for cn in self._control_nets:
                cn.unet.to(memory_format=torch.channels_last)
                cn.net.to(memory_format=torch.channels_last)

    def _set_embeddings(self):
        # Extended Textual Inversion および Textual Inversionを処理する
        if self.XTI_embeddings:
            diffusers.models.UNet2DConditionModel.forward = unet_forward_XTI
            diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D.forward = downblock_forward_XTI
            diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D.forward = upblock_forward_XTI
        
        if self.textual_inversion_embeddings:
            token_ids_embeds = []
            for embeds_file in self.textual_inversion_embeddings:
                if model_util.is_safetensors(embeds_file):
                    from safetensors.torch import load_file
                    data = load_file(embeds_file)
                else:
                    data = torch.load(embeds_file, map_location="cpu")

                if "string_to_param" in data:
                    data = data["string_to_param"]
                embeds = next(iter(data.values()))

                if type(embeds) != torch.Tensor:
                    raise ValueError(f"weight file does not contains Tensor / 重みファイルのデータがTensorではありません: {embeds_file}")

                num_vectors_per_token = embeds.size()[0]
                token_string = os.path.splitext(os.path.basename(embeds_file))[0]
                token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

                # add new word to tokenizer, count is num_vectors_per_token
                num_added_tokens = self._tokenizer.add_tokens(token_strings)
                if num_added_tokens != num_vectors_per_token:
                    # 会重复加载tokens
                    return
                # assert (
                #     num_added_tokens == num_vectors_per_token
                # ), f"tokenizer has same word to token string (filename). please rename the file / 指定した名前（ファイル名）のトークンが既に存在します。ファイルをリネームしてください: {embeds_file}"

                token_ids = self._tokenizer.convert_tokens_to_ids(token_strings)
                print(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids}")
                assert (
                    min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1
                ), f"token ids is not ordered"
                assert len(self._tokenizer) - 1 == token_ids[-1], f"token ids is not end of tokenize: {len(self._tokenizer)}"

                if num_vectors_per_token > 1:
                    self._pipe.add_token_replacement(token_ids[0], token_ids)

                token_ids_embeds.append((token_ids, embeds))

            self._text_encoder.resize_token_embeddings(len(self._tokenizer))
            token_embeds = self._text_encoder.get_input_embeddings().weight.data
            for token_ids, embeds in token_ids_embeds:
                for token_id, embed in zip(token_ids, embeds):
                    token_embeds[token_id] = embed
        
        if self.XTI_embeddings:
            XTI_layers = [
                "IN01",
                "IN02",
                "IN04",
                "IN05",
                "IN07",
                "IN08",
                "MID",
                "OUT03",
                "OUT04",
                "OUT05",
                "OUT06",
                "OUT07",
                "OUT08",
                "OUT09",
                "OUT10",
                "OUT11",
            ]
            token_ids_embeds_XTI = []
            for embeds_file in self.XTI_embeddings:
                if model_util.is_safetensors(embeds_file):
                    from safetensors.torch import load_file

                    data = load_file(embeds_file)
                else:
                    data = torch.load(embeds_file, map_location="cpu")
                if set(data.keys()) != set(XTI_layers):
                    raise ValueError("NOT XTI")
                embeds = torch.concat(list(data.values()))
                num_vectors_per_token = data["MID"].size()[0]

                token_string = os.path.splitext(os.path.basename(embeds_file))[0]
                token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

                # add new word to tokenizer, count is num_vectors_per_token
                num_added_tokens = self._tokenizer.add_tokens(token_strings)
                assert (
                    num_added_tokens == num_vectors_per_token
                ), f"tokenizer has same word to token string (filename). please rename the file / 指定した名前（ファイル名）のトークンが既に存在します。ファイルをリネームしてください: {embeds_file}"

                token_ids = self._tokenizer.convert_tokens_to_ids(token_strings)
                print(f"XTI embeddings `{token_string}` loaded. Tokens are added: {token_ids}")

                # if num_vectors_per_token > 1:
                self._pipe.add_token_replacement(token_ids[0], token_ids)

                token_strings_XTI = []
                for layer_name in XTI_layers:
                    token_strings_XTI += [f"{t}_{layer_name}" for t in token_strings]
                self._tokenizer.add_tokens(token_strings_XTI)
                token_ids_XTI = self._tokenizer.convert_tokens_to_ids(token_strings_XTI)
                token_ids_embeds_XTI.append((token_ids_XTI, embeds))
                for t in token_ids:
                    t_XTI_dic = {}
                    for i, layer_name in enumerate(XTI_layers):
                        t_XTI_dic[layer_name] = t + (i + 1) * num_added_tokens
                    self._pipe.add_token_replacement_XTI(t, t_XTI_dic)

                self._text_encoder.resize_token_embeddings(len(self._tokenizer))
                token_embeds = self._text_encoder.get_input_embeddings().weight.data
                for token_ids, embeds in token_ids_embeds_XTI:
                    for token_id, embed in zip(token_ids, embeds):
                        token_embeds[token_id] = embed

    def _set_pre_image(self, prompt_list):
        init_images = None
        if self.image_path is not None:
            print(f"load image for img2img: {self.image_path}")
            init_images = load_images(self.image_path)
            assert len(init_images) > 0, f"No image / 画像がありません: {self.image_path}"
            print(f"loaded {len(init_images)} images for img2img")

        mask_images = None
        if self.mask_path is not None:
            print(f"load mask for inpainting: {self.mask_path}")
            mask_images = self.load_images(self.mask_path)
            assert len(mask_images) > 0, f"No mask image / マスク画像がありません: {self.image_path}"
            print(f"loaded {len(mask_images)} mask images for inpainting")
        
        # promptがないとき、画像のPngInfoから取得する
        if init_images is not None and len(prompt_list) == 0:
            print("get prompts from images' meta data")
            for img in init_images:
                if "prompt" in img.text:
                    prompt = img.text["prompt"]
                    if "negative-prompt" in img.text:
                        prompt += " --n " + img.text["negative-prompt"]
                    prompt_list.append(prompt)

            # プロンプトと画像を一致させるため指定回数だけ繰り返す（画像を増幅する）
            l = []
            for im in init_images:
                l.extend([im] * self.images_per_prompt)
            init_images = l

            if mask_images is not None:
                l = []
                for im in mask_images:
                    l.extend([im] * self.images_per_prompt)
                mask_images = l

        # 画像サイズにオプション指定があるときはリサイズする
        if self.width is not None and self.height is not None:
            if init_images is not None:
                print(f"resize img2img source images to {self.width}*{self.height}")
                init_images = resize_images(init_images, (self.width, self.height))
            if mask_images is not None:
                print(f"resize img2img mask images to {self.width}*{self.height}")
                mask_images = resize_images(mask_images, (self.width, self.height))

        if mask_images:
            # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
            print("use mask as region")

            size = None
            for i, network in enumerate(self._networks):
                if i < 3:
                    np_mask = np.array(self._mask_images[0])
                    np_mask = np_mask[:, :, i]
                    size = np_mask.shape
                else:
                    np_mask = np.full(size, 255, dtype=np.uint8)
                mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
                network.set_region(i, i == len(self._networks) - 1, mask)
            mask_images = None

        guide_images = None
        if self.guide_image_path is not None:
            print(f"load image for CLIP/VGG16/ControlNet guidance: {self.guide_image_path}")
            guide_images = []
            for p in self.guide_image_path:
                guide_images.extend(load_images(p))

            print(f"loaded {len(guide_images)} guide images for guidance")
            if len(guide_images) == 0:
                print(f"No guide image, use previous generated image. / ガイド画像がありません。直前に生成した画像を使います: {self.image_path}")
                guide_images = None

        self._init_images = init_images
        self._mask_images = mask_images
        self._guide_images = guide_images

    def _process_batch(self, batch: List[BatchData], highres_fix, highres_1st=False):
        batch_size = len(batch)
        max_embeddings_multiples = 1 if self.max_embeddings_multiples is None else self.max_embeddings_multiples
        
        # highres_fixの処理
        if highres_fix and not highres_1st:
            # 1st stageのバッチを作成して呼び出す：サイズを小さくして呼び出す
            is_1st_latent = self._upscaler.support_latents() if self._upscaler else self.highres_fix_latents_upscaling

            print("process 1st stage")
            batch_1st = []
            for _, base, ext in batch:
                width_1st = int(ext.width * self.highres_fix_scale + 0.5)
                height_1st = int(ext.height * self.highres_fix_scale + 0.5)
                width_1st = width_1st - width_1st % 32
                height_1st = height_1st - height_1st % 32

                ext_1st = BatchDataExt(
                    width_1st,
                    height_1st,
                    self.highres_fix_steps,
                    ext.scale,
                    ext.negative_scale,
                    ext.strength,
                    ext.network_muls,
                    ext.num_sub_prompts,
                )
                batch_1st.append(BatchData(is_1st_latent, base, ext_1st))
            images_1st = self._process_batch(batch_1st, True, True)

            # 2nd stageのバッチを作成して以下処理する
            print("process 2nd stage")
            width_2nd, height_2nd = batch[0].ext.width, batch[0].ext.height

            if self._upscaler:
                # upscalerを使って画像を拡大する
                lowreso_imgs = None if is_1st_latent else images_1st
                lowreso_latents = None if not is_1st_latent else images_1st

                # 戻り値はPIL.Image.Imageかtorch.Tensorのlatents
                batch_size = len(images_1st)
                vae_batch_size = (
                    batch_size
                    if self.vae_batch_size is None
                    else (max(1, int(batch_size * self.vae_batch_size)) if self.vae_batch_size < 1 else self.vae_batch_size)
                )
                vae_batch_size = int(vae_batch_size)
                images_1st = self._upscaler.upscale(
                    self._vae, lowreso_imgs, lowreso_latents, self._dtype, width_2nd, height_2nd, batch_size, vae_batch_size
                )

            elif self.highres_fix_latents_upscaling:
                # latentを拡大する
                org_dtype = images_1st.dtype
                if images_1st.dtype == torch.bfloat16:
                    images_1st = images_1st.to(torch.float)  # interpolateがbf16をサポートしていない
                images_1st = torch.nn.functional.interpolate(
                    images_1st, (batch[0].ext.height // 8, batch[0].ext.width // 8), mode="bilinear"
                )  # , antialias=True)
                images_1st = images_1st.to(org_dtype)

            else:
                # 画像をLANCZOSで拡大する
                images_1st = [image.resize((width_2nd, height_2nd), resample=PIL.Image.LANCZOS) for image in images_1st]

            batch_2nd = []
            for i, (bd, image) in enumerate(zip(batch, images_1st)):
                bd_2nd = BatchData(False, BatchDataBase(*bd.base[0:3], bd.base.seed + 1, image, None, *bd.base[6:]), bd.ext)
                batch_2nd.append(bd_2nd)
            batch = batch_2nd

        # このバッチの情報を取り出す
        (
            return_latents,
            (step_first, _, _, _, init_image, mask_image, _, guide_image),
            (width, height, steps, scale, negative_scale, strength, network_muls, num_sub_prompts),
        ) = batch[0]
        noise_shape = (LATENT_CHANNELS, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR)

        prompts = []
        negative_prompts = []
        start_code = torch.zeros((batch_size, *noise_shape), device=self._device, dtype=self._dtype)
        noises = [
            torch.zeros((batch_size, *noise_shape), device=self._device, dtype=self._dtype)
            for _ in range(steps * self._scheduler_num_noises_per_step)
        ]
        seeds = []
        clip_prompts = []

        if init_image is not None:  # img2img?
            i2i_noises = torch.zeros((batch_size, *noise_shape), device=self._device, dtype=self._dtype)
            init_images = []

            if mask_image is not None:
                mask_images = []
            else:
                mask_images = None
        else:
            i2i_noises = None
            init_images = None
            mask_images = None

        if guide_image is not None:  # CLIP image guided?
            guide_images = []
        else:
            guide_images = None

        # バッチ内の位置に関わらず同じ乱数を使うためにここで乱数を生成しておく。あわせてimage/maskがbatch内で同一かチェックする
        all_images_are_same = True
        all_masks_are_same = True
        all_guide_images_are_same = True
        for i, (_, (_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image), _) in enumerate(batch):
            prompts.append(prompt)
            negative_prompts.append(negative_prompt)
            seeds.append(seed)
            clip_prompts.append(clip_prompt)

            if init_image is not None:
                init_images.append(init_image)
                if i > 0 and all_images_are_same:
                    all_images_are_same = init_images[-2] is init_image

            if mask_image is not None:
                mask_images.append(mask_image)
                if i > 0 and all_masks_are_same:
                    all_masks_are_same = mask_images[-2] is mask_image

            if guide_image is not None:
                if type(guide_image) is list:
                    guide_images.extend(guide_image)
                    all_guide_images_are_same = False
                else:
                    guide_images.append(guide_image)
                    if i > 0 and all_guide_images_are_same:
                        all_guide_images_are_same = guide_images[-2] is guide_image

            # make start code
            torch.manual_seed(seed)
            start_code[i] = torch.randn(noise_shape, device=self._device, dtype=self._dtype)

            # make each noises
            for j in range(steps * self._scheduler_num_noises_per_step):
                noises[j][i] = torch.randn(noise_shape, device=self._device, dtype=self._dtype)

            if i2i_noises is not None:  # img2img noise
                i2i_noises[i] = torch.randn(noise_shape, device=self._device, dtype=self._dtype)

        self._noise_manager.reset_sampler_noises(noises)

        # すべての画像が同じなら1枚だけpipeに渡すことでpipe側で処理を高速化する
        if init_images is not None and all_images_are_same:
            init_images = init_images[0]
        if mask_images is not None and all_masks_are_same:
            mask_images = mask_images[0]
        if guide_images is not None and all_guide_images_are_same:
            guide_images = guide_images[0]

        # ControlNet使用時はguide imageをリサイズする
        if self._control_nets:
            # TODO resampleのメソッド
            guide_images = guide_images if type(guide_images) == list else [guide_images]
            guide_images = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in guide_images]
            if len(guide_images) == 1:
                guide_images = guide_images[0]

        # generate
        if self._networks:
            shared = {}
            for n, m in zip(self._networks, network_muls if network_muls else self._network_default_muls):
                n.set_multiplier(m)
                if self._mask_images:
                    n.set_current_generation(batch_size, num_sub_prompts, width, height, shared)

        images = self._pipe(
            prompts,
            negative_prompts,
            init_images,
            mask_images,
            height,
            width,
            steps,
            scale,
            negative_scale,
            strength,
            latents=start_code,
            output_type="pil",
            max_embeddings_multiples=max_embeddings_multiples,
            img2img_noise=i2i_noises,
            vae_batch_size=self.vae_batch_size,
            return_latents=return_latents,
            clip_prompts=clip_prompts,
            clip_guide_images=guide_images,
        )[0]
        
        if highres_1st and not self.highres_fix_save_1st:  # return images or latents
            return images

        # save image
        highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        for i, (image, prompt, negative_prompts, seed, clip_prompt) in enumerate(
            zip(images, prompts, negative_prompts, seeds, clip_prompts)
        ):
            metadata = PngInfo()
            metadata.add_text("prompt", prompt)
            metadata.add_text("seed", str(seed))
            metadata.add_text("sampler", self.sampler)
            metadata.add_text("steps", str(steps))
            metadata.add_text("scale", str(scale))
            if negative_prompt is not None:
                metadata.add_text("negative-prompt", negative_prompt)
            if negative_scale is not None:
                metadata.add_text("negative-scale", str(negative_scale))
            if clip_prompt is not None:
                metadata.add_text("clip-prompt", clip_prompt)

            if self.use_original_file_name and init_images is not None:
                if type(init_images) is list:
                    fln = os.path.splitext(os.path.basename(init_images[i % len(init_images)].filename))[0] + ".png"
                else:
                    fln = os.path.splitext(os.path.basename(init_images.filename))[0] + ".png"
                if self.img_name_prefix:
                    fln =  f"{self.img_name_prefix}-{fln}"
            elif self.sequential_file_name:
                if self.img_name_prefix:
                    fln = f"{self.img_name_prefix}-{highres_prefix}{step_first + i + 1:06d}.png"
                else:
                    fln = f"im_{highres_prefix}{step_first + i + 1:06d}.png"
            else:
                if self.img_name_prefix:
                    fln = f"{self.img_name_prefix}-{highres_prefix}{i:03d}.png"
                else:
                    fln = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"

            
            image.save(os.path.join(self.outdir, fln), pnginfo=metadata)

        return images

    def gen_batch_process(self):
        self._set_embeddings()

        # promptを取得する
        prompt_list = get_prompt_list(self.prompt, self.from_file)
 
        self._set_pre_image(prompt_list)

        # seed指定時はseedを決めておく
        if self.seed != -1:
            random.seed(self.seed)
            predefined_seeds = [random.randint(0, 0x7FFFFFFF) for _ in range(self.n_iter * len(prompt_list) * self.images_per_prompt)]
            if len(predefined_seeds) == 1:
                predefined_seeds[0] = self.seed
        else:
            predefined_seeds = None

        # 画像生成のループ
        os.makedirs(self.outdir, exist_ok=True)
        
        prev_image = None  # for VGG16 guided
        for gen_iter in range(self.n_iter):
            print(f"iteration {gen_iter+1}/{self.n_iter}")
            iter_seed = random.randint(0, 0x7FFFFFFF)

            # 画像生成のプロンプトが一周するまでのループ
            prompt_index = 0
            global_step = 0
            batch_data = []
            while prompt_index < len(prompt_list):
                if len(prompt_list) == 0:
                    valid = False
                    while not valid:
                        print("\nType prompt:")
                        try:
                            prompt = input()
                        except EOFError:
                            break

                        valid = len(prompt.strip().split(" --")[0].strip()) > 0
                    if not valid:  # EOF, end app
                        break
                else:
                    prompt = prompt_list[prompt_index]

                # parse prompt
                width = self.width  
                height = self.height
                scale = self.scale
                negative_scale = self.negative_scale
                steps = self.steps
                seeds = None
                strength = 0.8 if self.strength is None else self.strength
                negative_prompt = ""
                clip_prompt = None
                network_muls = None
                
                prompt_args = prompt.strip().split(" --")
                prompt = prompt_args[0]
                print(f"prompt {prompt_index+1}/{len(prompt_list)}: {prompt}")

                for parg in prompt_args[1:]:
                    try:
                        m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                        if m:
                            width = int(m.group(1))
                            print(f"width: {width}")
                            continue

                        m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                        if m:
                            height = int(m.group(1))
                            print(f"height: {height}")
                            continue

                        m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                        if m:  # steps
                            steps = max(1, min(1000, int(m.group(1))))
                            print(f"steps: {steps}")
                            continue

                        m = re.match(r"d ([\d,]+)", parg, re.IGNORECASE)
                        if m:  # seed
                            seeds = [int(d) for d in m.group(1).split(",")]
                            print(f"seeds: {seeds}")
                            continue

                        m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                        if m:  # scale
                            scale = float(m.group(1))
                            print(f"scale: {scale}")
                            continue

                        m = re.match(r"nl ([\d\.]+|none|None)", parg, re.IGNORECASE)
                        if m:  # negative scale
                            if m.group(1).lower() == "none":
                                negative_scale = None
                            else:
                                negative_scale = float(m.group(1))
                            print(f"negative scale: {negative_scale}")
                            continue

                        m = re.match(r"t ([\d\.]+)", parg, re.IGNORECASE)
                        if m:  # strength
                            strength = float(m.group(1))
                            print(f"strength: {strength}")
                            continue

                        m = re.match(r"n (.+)", parg, re.IGNORECASE)
                        if m:  # negative prompt
                            negative_prompt = m.group(1)
                            print(f"negative prompt: {negative_prompt}")
                            continue

                        m = re.match(r"c (.+)", parg, re.IGNORECASE)
                        if m:  # clip prompt
                            clip_prompt = m.group(1)
                            print(f"clip prompt: {clip_prompt}")
                            continue

                        m = re.match(r"am ([\d\.\-,]+)", parg, re.IGNORECASE)
                        if m:  # network multiplies
                            network_muls = [float(v) for v in m.group(1).split(",")]
                            while len(network_muls) < len(self._networks):
                                network_muls.append(network_muls[-1])
                            print(f"network mul: {network_muls}")
                            continue

                    except ValueError as ex:
                        print(f"Exception in parsing / 解析エラー: {parg}")
                        print(ex)

                seeds = self._gen_seeds(predefined_seeds, iter_seed, seeds)

                init_image = mask_image = guide_image = None
                for seed in seeds:  # images_per_promptの数だけ
                    # 同一イメージを使うとき、本当はlatentに変換しておくと無駄がないが面倒なのでとりあえず毎回処理する
                    if self._init_images is not None:
                        init_image = self._init_images[global_step % len(self._init_images)]

                        # 32単位に丸めたやつにresizeされるので踏襲する
                        width, height = init_image.size
                        width = width - width % 32
                        height = height - height % 32
                        if width != init_image.size[0] or height != init_image.size[1]:
                            print(
                                f"img2img image size is not divisible by 32 so aspect ratio is changed / img2imgの画像サイズが32で割り切れないためリサイズされます。画像が歪みます"
                            )

                    if self._mask_images is not None:
                        mask_image = self._mask_images[global_step % len(self._mask_images)]

                    if self._guide_images is not None:
                        if self._control_nets:  # 複数件の場合あり
                            c = len(self._control_nets)
                            p = global_step % (len(self._guide_images) // c)
                            guide_image = self._guide_images[p * c : p * c + c]
                        else:
                            guide_image = self._guide_images[global_step % len(self._guide_images)]
                    elif self.clip_image_guidance_scale > 0 or self.vgg16_guidance_scale > 0:
                        if prev_image is None:
                            print("Generate 1st image without guide image.")
                        else:
                            print("Use previous image as guide image.")
                            guide_image = prev_image

                    if self._mask_images:
                        num_sub_prompts = len(prompt.split(" AND "))
                        assert (
                            len(self._networks) <= num_sub_prompts
                        ), "Number of networks must be less than or equal to number of sub prompts."
                    else:
                        num_sub_prompts = None

                    b1 = BatchData(
                        False,
                        BatchDataBase(global_step, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image),
                        BatchDataExt(
                            width,
                            height,
                            steps,
                            scale,
                            negative_scale,
                            strength,
                            tuple(network_muls) if network_muls else None,
                            num_sub_prompts,
                        ),
                    )
                    if len(batch_data) > 0 and batch_data[-1].ext != b1.ext:  # バッチ分割必要？
                        self._process_batch(batch_data, self.highres_fix_scale)
                        batch_data.clear()

                    batch_data.append(b1)
                    if len(batch_data) == self.batch_size:
                        prev_image = self._process_batch(batch_data, self.highres_fix_scale)[0]
                        batch_data.clear()

                    global_step += 1

                prompt_index += 1

            if len(batch_data) > 0:
                self._process_batch(batch_data, self.highres_fix_scale)
                batch_data.clear()

        print("done!")

    def _gen_seeds(self, predefined_seeds, iter_seed, seeds):
        if seeds is not None:
                    # 数が足りないなら繰り返す
            if len(seeds) < self.images_per_prompt:
                seeds = seeds * int(math.ceil(self.images_per_prompt / len(seeds)))
            seeds = seeds[: self.images_per_prompt]
        else:
            if predefined_seeds is not None:
                seeds = predefined_seeds[-self.images_per_prompt :]
                predefined_seeds = predefined_seeds[: -self.images_per_prompt]
            elif self.iter_same_seed:
                seeds = [iter_seed] * self.images_per_prompt
            else:
                seeds = [random.randint(0, 0x7FFFFFFF) for _ in range(self.images_per_prompt)]
        return seeds
    
    def gen_once_image(self):
        self.create_pipline()

        self.load_network()

        self.gen_batch_process()

