import sys, os 
import random

project_path = os.path.abspath(".")
sys.path.append(os.path.join(project_path, "points", "diffusion_generator"))
sys.path.append(os.path.join(project_path, "points", "datasets"))

from datetime import datetime
from diffusion_generator import image_generator

class TrainProject:
   def __init__(self, train_name, train_repeat) -> None:
      self.base_model = os.path.join(project_path, "models" , "NAI-full.ckpt")
      self.train_name = train_name
      self.train_repeat = train_repeat
      self.work_path = os.path.abspath(".")
      self.project_path = None
      self.train_img_path = None
      self.tagger_img_path = None
      self.sources_img_path = None
      self.checkpoints_path = None
      self.log_path = None
      self.train_config_path = None
   
   def init_project(self, dt=None):
      global project_path
      pn_path = os.path.join(project_path, "train-projects" , self.train_name)
      if not os.path.exists(pn_path):
         os.makedirs(pn_path)
      if dt is None:
         dt = datetime.now().strftime('%Y%m%d%H%M%S')
      pt_path = os.path.join(pn_path, "train-%s" % dt)
      if not os.path.exists(pt_path):
         os.makedirs(pt_path)

      self.project_path = pt_path
      self.sources_img_path = os.path.join(self.work_path, "resources" , "source-images")
      self.checkpoints_path = os.path.join(self.project_path, "checkpoints")
      self.log_path = os.path.join(self.project_path, "logs")
      self.train_config_path = os.path.join(project_path, "resources", "train_config.json")

   def test_cmd_checkpoints(self):
      """
      通过sd_script命令进行生成，由于通过命令，每一次生成都需要重新加载模型
      https://note.com/kohya_ss/n/n2693183a798e

      single LoRA:
      python gen_img_diffusers.py --ckpt name_to_ckpt.safetensors --n_iter 1 --scale 8 --steps 40 \
         --outdir txt2img/samples --xformers --W 512 --H 512 --fp16 --sampler k_euler_a \
         --network_module networks.lora --network_weights lora1.safetensors --network_mul 1.0 \
         --max_embeddings_multiples 3 --clip_skip 2 --batch_size 1 --images_per_prompt 1 \
         --prompt "beautiful scene --n negative prompt"

      two LoRAs:
      python gen_img_diffusers.py --ckpt name_to_ckpt.safetensors --n_iter 1 --scale 8 --steps 40 \
         --outdir txt2img/samples --xformers --W 512 --H 512 --fp16 --sampler k_euler_a \
         --network_module networks.lora networks.lora --network_weights lora1.safetensors lora2.safetensors --network_mul 1.0 0.8 \
         --max_embeddings_multiples 3 --clip_skip 2 --batch_size 1 --images_per_prompt 1 \
         --prompt "beautiful scene --n negative prompt"
      """

      """
      https://github.com/kohya-ss/sd-webui-additional-networks/issues/148

      python .\points\sd_scripts\gen_img_diffusers.py --prompt "1girl" --outdir E:/Develop/sd-train-links/outputs ^
         --H 512 --W 512 --steps 30 --sampler k_dpm_2 --ckpt E:/Develop/sd-train-links/models/NAI-full.ckpt ^
         --network_module networks.lora --network_weights E:/Develop/sd-train-links/outputs/dribbble-design-000001.safetensors ^
         --network_mul 1.0 --seed -1 --clip_skip 2 --fp16 --xformers
      """
      seed = random.randint(0, 0x7FFFFFFF)
      gen_dict = {
         "ckpt" : self.base_model,
         "sampler" : "dpmsolver++",
         "max_embeddings_multiples" : 3,
         "clip_skip": 2,
         "seed" : seed,
         "fp16": True,
         "xformers": True,
      }

      check_points = [name for name in os.listdir(self.checkpoints_path)if name.endswith('.safetensors')]

      for check_point in check_points:                                         
         args_list = []
         for key, value in gen_dict.items():
            if not value:
               continue
            if isinstance(value, bool):
               args_list.append(f"--{key}")
            else:
               args_list.append(f"--{key}={value}")
         args_list.append(f"--network_module=networks.lora")
         args_list.append(f"--network_weights=%s" % os.path.join(self.checkpoints_path, check_point))
         args_list.append(f"--network_mul=0.6")
         args_list.append(f"--sequential_file_name")
         args_list.append(f"--prompt=masterpiece, best quality, 1girl" + " --n " + "bad quality, worst quality, bad anatomy, bad hands")
         name_space = image_generator.setup_parser().parse_args(args_list)
         image_generator.main(name_space)
   
   def test_cmd_checkpoints_once(self):
      import subprocess
      args = []
      args.append('--prompt "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, (smile:1.1),(closed mouth) medium breasts,beautiful detailed eyes,(collared shirt:1.1),pleated skirt,(long hair:1.2),floating hair --n EasyNegative"')
      args.append("--H 768")
      args.append("--W 512")
      args.append("--steps 28")
      args.append("--fp16")
      args.append("--ckpt %s" % os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
      args.append("--outdir %s" % os.path.join(self.project_path, "images"))
      args.append("--sampler dpmsolver++")
      args.append("--clip_skip 2")
      args.append("--scale 8")
      args.append("--seed 280681258")
      args.append("--xformers")
      args.append("--max_embeddings_multiples 3")
      args.append("--textual_inversion_embeddings %s" % os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors"))
      python = r"python310\python.exe"
      cmd = f"{python} points/sd_scripts/gen_img_diffusers.py %s" % " ".join(args)
      print(cmd)
      subprocess.check_call(cmd.split(" "))

   def test_checkpoints_once(self):
      gen_image = image_generator.GenImages()
      gen_image.set_dtype("fp16")
      gen_image.set_ckpt(os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
      gen_image.outdir = os.path.join(self.project_path, "images")
      gen_image.steps = 28
      gen_image.height = 768
      gen_image.width = 512
      gen_image.sampler = "dpmsolver++"
      gen_image.clip_skip = 2
      gen_image.scale = 8
      gen_image.seed = 280681258
      gen_image.xformers = True
      gen_image.max_embeddings_multiples = 3
      gen_image.textual_inversion_embeddings = [os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")]
      
      # gen_image.network_module = ["networks.lora"]
      # gen_image.network_weights = [os.path.join(self.checkpoints_path, "dribbble-design-000001.safetensors")]
      # gen_image.network_mul = [0.6]
      gen_image.prompt = "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, (smile:1.1),(closed mouth) medium breasts,beautiful detailed eyes,(collared shirt:1.1),pleated skirt,(long hair:1.2),floating hair --n EasyNegative"
      gen_image.gen_once_image()

   def test_checkpoints_n(self):
      seed = random.randint(0, 0x7FFFFFFF)
      self.checkpoints_path = "E:\\ai-stable-diffsuion\\LoRA\\lora-train\\outputs\\ghiblistyle\\\dylora"

      check_points = [name for name in os.listdir(self.checkpoints_path)if name.endswith('.safetensors')]

      gen_image = image_generator.GenImages()
      gen_image.set_dtype("fp16")
      gen_image.set_ckpt(os.path.join(project_path, "models" , "NAI-full.ckpt"))
      gen_image.outdir = os.path.join(self.project_path, "images")
      gen_image.steps = 30
      gen_image.sampler = "dpmsolver++"
      gen_image.clip_skip = 2
      gen_image.width = 512
      gen_image.height = 768
      gen_image.seed = seed
      gen_image.xformers = True
      gen_image.max_embeddings_multiples = 3
      gen_image.img_name_type = "network"
      
      gen_image.create_pipline()
      for check_point in check_points:               
         gen_image.network_module = ["networks.lora"]
         gen_image.network_weights = [os.path.join(self.checkpoints_path, check_point)]
         gen_image.network_mul = [1]
         gen_image.textual_inversion_embeddings = [os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")]
         gen_image.load_network(append_network=False)

         gen_image.prompt = "masterpiece, best quality, 1girl, food, short hair, solo, indoors, suspenders, shirt, rice, holding, open mouth, short sleeves, yellow shirt, sitting, smile, black hair, barefoot, apron, chopsticks, black eyes, tatami, collared shirt, seiza, child, bowl" + " --n EasyNegative"
         gen_image.gen_batch_process()


if __name__ == '__main__':
   train_name = "dribbble-design"
   train_repeat = 8

   design_project = TrainProject("dribbble-design", 8)
   
   design_project.init_project()
   design_project.test_checkpoints_n()
