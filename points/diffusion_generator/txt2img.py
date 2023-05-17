from typing import List
import glob
import importlib
import time
import os
import random
import sys

import diffusers
import torch

project_path = os.path.abspath(".")
sys.path.append(os.path.join(project_path, "points", "diffusion_generator"))
sys.path.append(os.path.join(project_path, "points", "datasets"))

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
from PIL.PngImagePlugin import PngInfo


class Txt2Img():
    def __init__(self):
         # load Stable Diffusion v2.0 model
        self.v2 = False 
        # enable v-parameterization training
        self.v_parameterization = False

        # directory for caching Tokenizer (for offline training)
        self.tokenizer_cache_dir = None
        self.max_embeddings_multiples = 3
        self.vae_batch_size = 0
        self.diffusers_xformers = True

        self.outdir = None

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

        self._scheduler_num_noises_per_step = 0

        self._dtype = None
        self._ckpt = None

        self._text_encodert = None
        self._vaet = None
        self._unet = None
        self._scheduler = None
        self._noise_manager = None

    def set_dtype(self, dtype:str=["fp16" ,"bp16", "fp32"]):
        if dtype == "fp16":
            self._dtype = torch.float16
        elif dtype == "bp16":
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float32

    def load_ckpt(self, ckpt_path:str):
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

        if self.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            print(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        self._tokenizer = tokenizer
    
    def _load_scheduler(self, sampler):
        sched_init_args = {}
        self._scheduler_num_noises_per_step = 1
        if sampler == "ddim":
            scheduler_cls = DDIMScheduler
            scheduler_module = diffusers.schedulers.scheduling_ddim
        elif sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
            scheduler_cls = DDPMScheduler
            scheduler_module = diffusers.schedulers.scheduling_ddpm
        elif sampler == "pndm":
            scheduler_cls = PNDMScheduler
            scheduler_module = diffusers.schedulers.scheduling_pndm
        elif sampler == "lms" or sampler == "k_lms":
            scheduler_cls = LMSDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_lms_discrete
        elif sampler == "euler" or sampler == "k_euler":
            scheduler_cls = EulerDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_euler_discrete
        elif sampler == "euler_a" or sampler == "k_euler_a":
            scheduler_cls = EulerAncestralDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
        elif sampler == "dpmsolver" or sampler == "dpmsolver++":
            scheduler_cls = DPMSolverMultistepScheduler
            sched_init_args["algorithm_type"] = sampler
            scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep
        elif sampler == "dpmsingle":
            scheduler_cls = DPMSolverSinglestepScheduler
            scheduler_module = diffusers.schedulers.scheduling_dpmsolver_singlestep
        elif sampler == "heun":
            scheduler_cls = HeunDiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_heun_discrete
        elif sampler == "dpm_2" or sampler == "k_dpm_2":
            scheduler_cls = KDPM2DiscreteScheduler
            scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_discrete
        elif sampler == "dpm_2_a" or sampler == "k_dpm_2_a":
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
    

    def create_pipline(self, clip_skip=1, sampler="ddim"):
        if self.v_parameterization and not self.v2:
            print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
        if self.v2 and clip_skip > 1:
            print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

        # xformers、Hypernetwork 一致
        if not self.diffusers_xformers:
            replace_unet_modules(self._unet, not self.xformers, self.xformers)

        # load tokenizer token分词器
        self._load_tokenizer()

        # load scheduler 扩散调度器
        self._load_scheduler(sampler)

        # 确定 设备类型，使用GPU or CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps"を考量してない

        # custom pipeline
        self._vae.to(self._dtype).to(self._device)
        self._text_encoder.to(self._dtype).to(self._device)
        self._unet.to(self._dtype).to(self._device)
        
        # set highres
        self._set_highres()

        self._pipe = PipelineLike(
            self._device,
            self._vae,
            self._text_encoder,
            self._tokenizer,
            self._unet,
            self._scheduler,
            clip_skip,
            None, 0, 0, None, 0, 0
        )
        print("pipeline is ready.")

        if self.diffusers_xformers:
            self._pipe.enable_xformers_memory_efficient_attention()


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
            (step_first, _, _, _, _, _, _, _),
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

        # バッチ内の位置に関わらず同じ乱数を使うためにここで乱数を生成しておく。あわせてimage/maskがbatch内で同一かチェックする
        for i, (_, (_, prompt, negative_prompt, seed, _, _, _, _), _) in enumerate(batch):
            prompts.append(prompt)
            negative_prompts.append(negative_prompt)
            seeds.append(seed)

            # make start code
            torch.manual_seed(seed)
            start_code[i] = torch.randn(noise_shape, device=self._device, dtype=self._dtype)

            # make each noises
            for j in range(steps * self._scheduler_num_noises_per_step):
                noises[j][i] = torch.randn(noise_shape, device=self._device, dtype=self._dtype)

        self._noise_manager.reset_sampler_noises(noises)

        # # generate
        # if self._networks:
        #     shared = {}
        #     for n, m in zip(self._networks, network_muls if network_muls else self._network_default_muls):
        #         n.set_multiplier(m)
        #         if self._mask_images:
        #             n.set_current_generation(batch_size, num_sub_prompts, width, height, shared)

        images = self._pipe(
            prompts,
            negative_prompts,
            None,
            None,
            height,
            width,
            steps,
            scale,
            negative_scale,
            strength,
            latents=start_code,
            output_type="pil",
            max_embeddings_multiples=max_embeddings_multiples,
            img2img_noise=None,
            vae_batch_size=self.vae_batch_size,
            return_latents=return_latents,
            clip_prompts=[],
            clip_guide_images=None,
        )[0]
        
        if highres_1st and not self.highres_fix_save_1st:  # return images or latents
            return images

        # save image
        highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        for i, (image, prompt, negative_prompts, seed) in enumerate(
            zip(images, prompts, negative_prompts, seeds)
        ):
            metadata = PngInfo()
            metadata.add_text("prompt", prompt)
            metadata.add_text("seed", str(seed))
            # metadata.add_text("sampler", self.sampler)
            metadata.add_text("steps", str(steps))
            metadata.add_text("scale", str(scale))
            if negative_prompt is not None:
                metadata.add_text("negative-prompt", negative_prompt)
            if negative_scale is not None:
                metadata.add_text("negative-scale", str(negative_scale))

            # if self.sequential_file_name:
            #     if self.img_name_prefix:
            #         fln = f"{self.img_name_prefix}-{highres_prefix}{step_first + i + 1:06d}.png"
            #     else:
            #         fln = f"im_{highres_prefix}{step_first + i + 1:06d}.png"
            # else:
            #     if self.img_name_prefix:
            #         fln = f"{self.img_name_prefix}-{highres_prefix}{i:03d}.png"
            #     else:
            #         fln = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"

            fln = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"
            
            image.save(os.path.join(self.outdir, fln), pnginfo=metadata)

        return images

    def gen_batch_process(self, prompt, negative_prompt, width, height, steps, scale, negative_scale=None, strength=0.8, seed=-1, n_iter=1, batch_size=1):
        # self._set_embeddings()

        # seed
        if seed is None or seed == -1:
            seed = random.randint(0, 0x7FFFFFFF)

        # 画像生成のループ
        # os.makedirs(self.outdir, exist_ok=True)
        
        for gen_iter in range(n_iter):
            print(f"iteration {gen_iter+1}/{n_iter}")
            iter_seed = random.randint(0, 0x7FFFFFFF)

            # 画像生成のプロンプトが一周するまでのループ
            global_step = 0
            batch_data = []

            seeds=[seed]# seeds = self._gen_seeds(predefined_seeds, iter_seed, seeds)

            for seed in seeds:  # images_per_promptの数だけ
                b1 = BatchData(
                    False,
                    BatchDataBase(global_step, prompt, negative_prompt, seed, None, None, None, None),
                    BatchDataExt(
                           width,
                        height,
                        steps,
                        scale,
                        negative_scale,
                        strength,
                        None, # tuple(network_muls) if network_muls else None,
                        None,
                    ),
                )
                if len(batch_data) > 0 and batch_data[-1].ext != b1.ext:  # バッチ分割必要？
                    self._process_batch(batch_data, self.highres_fix_scale)
                    batch_data.clear()

                batch_data.append(b1)
                if len(batch_data) == batch_size:
                    prev_image = self._process_batch(batch_data, self.highres_fix_scale)[0]
                    batch_data.clear()

                global_step += 1

            if len(batch_data) > 0:
                self._process_batch(batch_data, self.highres_fix_scale)
                batch_data.clear()
    

if __name__ == '__main__':
    txt2Img = Txt2Img()
    txt2Img.set_dtype("fp16")
    txt2Img.xformers = True
    txt2Img.outdir = "E:/"

    txt2Img.load_ckpt(os.path.join(project_path, "models" , "NAI-full.ckpt"))
    txt2Img.create_pipline()
    txt2Img.gen_batch_process(
        "masterpiece, best quality, 1girl, food, short hair, solo, indoors, suspenders, shirt, rice, holding, open mouth, short sleeves, yellow shirt, sitting, smile, black hair, barefoot, apron, chopsticks, black eyes, tatami, collared shirt, seiza, child, bowl",
        "",
        512,
        512,
        30,
        7.5,
    )
