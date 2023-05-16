import os

from typing import List, Dict
from pathlib import Path

# from preload import default_ddp_path
# from tagger.preset import Preset
from interrogator import Interrogator, WaifuDiffusionInterrogator

# preset = Preset(Path(scripts.basedir(), 'presets'))

# interrogators: Dict[str, Interrogator] = {}
interrogators: Dict[str, Interrogator] = {
    'wd14-convnextv2-v2': WaifuDiffusionInterrogator(
        'wd14-convnextv2-v2',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
        revision='v2.0'
    ),
    'wd14-vit-v2': WaifuDiffusionInterrogator(
        'wd14-vit-v2',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2',
        revision='v2.0'
    ),
    'wd14-convnext-v2': WaifuDiffusionInterrogator(
        'wd14-convnext-v2',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2',
        revision='v2.0'
    ),
    'wd14-swinv2-v2': WaifuDiffusionInterrogator(
        'wd14-swinv2-v2',
        repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2',
        revision='v2.0'
    ),
    'wd14-convnextv2-v2-git': WaifuDiffusionInterrogator(
        'wd14-convnextv2-v2',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    ),
    'wd14-vit-v2-git': WaifuDiffusionInterrogator(
        'wd14-vit-v2-git',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2'
    ),
    'wd14-convnext-v2-git': WaifuDiffusionInterrogator(
        'wd14-convnext-v2-git',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2'
    ),
    'wd14-swinv2-v2-git': WaifuDiffusionInterrogator(
        'wd14-swinv2-v2-git',
        repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2'
    ),
    'wd14-vit': WaifuDiffusionInterrogator(
        'wd14-vit',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger'),
    'wd14-convnext': WaifuDiffusionInterrogator(
        'wd14-convnext',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger'
    ),
    'wd-v1-4-convnextv2-tagger-v2': WaifuDiffusionInterrogator(
        'wd-v1-4-convnextv2-tagger-v2',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2'
    ),
}


def get_data(name, repo_id, revision=''):
    return {
        'name': name,
        'repo_id': repo_id,
        'revistion': revision
    }


interrogators_data = {
    'wd14-convnextv2-v2': get_data(
        'wd14-convnextv2-v2',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
        revision='v2.0'
    ),
    'wd14-vit-v2': get_data(
        'wd14-vit-v2',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2',
        revision='v2.0'
    ),
    'wd14-convnext-v2': get_data(
        'wd14-convnext-v2',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2',
        revision='v2.0'
    ),
    'wd14-swinv2-v2': get_data(
        'wd14-swinv2-v2',
        repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2',
        revision='v2.0'
    ),
    'wd14-convnextv2-v2-git': get_data(
        'wd14-convnextv2-v2',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    ),
    'wd14-vit-v2-git': get_data(
        'wd14-vit-v2-git',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2'
    ),
    'wd14-convnext-v2-git': get_data(
        'wd14-convnext-v2-git',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2'
    ),
    'wd14-swinv2-v2-git': get_data(
        'wd14-swinv2-v2-git',
        repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2'
    ),
    'wd14-vit': get_data(
        'wd14-vit',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger'),
    'wd14-convnext': get_data(
        'wd14-convnext',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger'
    ),
    'wd-v1-4-convnextv2-tagger-v2': get_data(
        'wd-v1-4-convnextv2-tagger-v2',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2'
    ),
}

def new_interrogator(name, tf_device_name):
    data = interrogators_data[name]
    interrogator = WaifuDiffusionInterrogator(data['name'],  tf_device_name=tf_device_name, repo_id=data['repo_id'],)
    return interrogator

def split_str(s: str, separator=',') -> List[str]:
    return [x.strip() for x in s.split(separator) if x]
