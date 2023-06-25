import sys, os 
import shutil
import argparse
import random

project_path = os.path.abspath(".")
sys.path.append(os.path.join(project_path, "points", "diffusion_generator"))
sys.path.append(os.path.join(project_path, "points", "datasets"))

from datetime import datetime
from datasets.auto_tagging import tagger
from sd_scripts import train_network
from diffusion_generator import gen_img
from train_scripts.ArgsList import ArgStore
from train_scripts.Parser import Parser
from train_scripts import json_functions

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

   def build_datasets(self):
      self.train_img_path = os.path.join(self.project_path, self.train_name)
      self.tagger_img_path = os.path.join(self.train_img_path, "%d_%s"%(self.train_repeat,  self.train_name))
      if os.path.exists(self.tagger_img_path):
         pass
         # shutil.rmtree(tsimg_path)
      else:
         shutil.copytree(self.sources_img_path, self.tagger_img_path)
      tagger.on_interrogate(self.tagger_img_path, batch_output_dir=self.tagger_img_path)

   def clean_tags(self, tags=[]):
      pass


   def do_train(self):
      # 2. 训练
      args = ArgStore.convert_args_to_dict()
      
      args['base_model'] = self.base_model
      args['img_folder'] = self.train_img_path
      args['output_folder'] = self.checkpoints_path
      args['change_output_name'] = self.train_name
      args['log_dir'] = self.log_path

      # "sample_sampler": "dpmsolver++",
      args['sample_prompts'] = None
      args['sample_every_n_steps'] = None
      args['sample_every_n_epochs'] = None

      # json/toml?
      json_functions.load_json(self.train_config_path, args)
      args = Parser().create_args(ArgStore.change_dict_to_internal_names(args))
      # 2.2 补完json，也要复制一份做存档
      train_network.train(args)

   def gen_checkpoint_images(self):
      seed = random.randint(0, 0x7FFFFFFF)
      check_points = [name for name in os.listdir(self.checkpoints_path)if name.endswith('.safetensors')]

      gen_image = gen_img.GenImages()
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
   train_name = "gufeng"
   train_repeat = 8

   design_project = TrainProject("gufeng", 8)
   
   # 初始化项目
   design_project.init_project()

   # 构建数据集
   design_project.build_datasets()

   # 清理标签
   tags = []
   design_project.clean_tags(tags)

   # 开始训练
   design_project.do_train()

   # 生成实例图片
   design_project.gen_checkpoint_images()
