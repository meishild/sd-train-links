import sys, os 

project_path = os.path.abspath(".")
sys.path.append(os.path.join(project_path, "points", "datasets"))

from mldanbooru.interface import Infer

if __name__ == '__main__':
    dir = os.path.join(project_path, "resources" , "source-images")
    infer = Infer()
    infer.infer_folder(
        path=os.path.join(project_path, "resources", "source-images"),
        threshold=0.7, # 置信度
        image_size=448, 
        keep_ratio=True,
        model_name="ml_caformer_m36_fp16_dec-5-97527.ckpt",
        space=False, # 使用_替换空格
        escape=False, # 使用文本转译特殊符号
        out_type="txt")