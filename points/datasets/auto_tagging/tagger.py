import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import json
import traceback

from collections import OrderedDict
from pathlib import Path
from glob import glob
from PIL import Image, UnidentifiedImageError

import utils
from utils import split_str
from interrogator import Interrogator
import format

def on_interrogate(
        batch_input_glob: str,
        batch_output_dir: str = '',
        batch_output_filename_format: str = '[name].[output_extension]',
        batch_output_action_on_conflict: str = 'ignore',  # prepend ignore copy 'add'
        batch_remove_duplicated_tag: bool = True,
        batch_output_save_json: bool = False,

        interrogator: str = 'wd14-vit-v2-git',
        threshold: float = 0.2,
        additional_tags: str = '',
        exclude_tags: str = '',
        sort_by_alphabetical_order: bool = False,
        add_confident_as_weight: bool = False,
        replace_underscore: bool = False,
        replace_underscore_excludes: str = '',
        escape_tag: bool = False,

        unload_model_after_running: bool = False,
        target_paths = None,
        tf_device_name = 'cpu:0',
        logger = None
):
    if logger:
        logger.log(0.00, 'downloading model')
    # utils.refresh_interrogators()
    if interrogator not in utils.interrogators:
        raise TypeError(f"'{interrogator}' is not a valid interrogator({utils.interrogators})")
        # return ['', None, None, f"'{interrogator}' is not a valid interrogator"]

    interrogator: Interrogator = utils.new_interrogator(interrogator, tf_device_name)

    postprocess_opts = (
        threshold,
        split_str(additional_tags),
        split_str(exclude_tags),
        sort_by_alphabetical_order,
        add_confident_as_weight,
        replace_underscore,
        split_str(replace_underscore_excludes),
        escape_tag
    )
    if logger:
        logger.log(0.01, 'loading model')
    # batch process
    batch_input_glob = batch_input_glob.strip()
    batch_output_dir = batch_output_dir.strip()
    batch_output_filename_format = batch_output_filename_format.strip()

    if batch_input_glob != '':
        # if there is no glob pattern, insert it automatically
        if not batch_input_glob.endswith('*'):
            if not batch_input_glob.endswith(os.sep):
                batch_input_glob += os.sep
            batch_input_glob += '*'

        # get root directory of input glob pattern
        base_dir = batch_input_glob.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)

        # check the input directory path
        if not os.path.isdir(base_dir):
            raise TypeError(f'input path{base_dir} is not a directory')
            # return ['', None, None, 'input path is not a directory']

        # this line is moved here because some reason
        # PIL.Image.registered_extensions() returns only PNG if you call too early
        supported_extensions = [
            e
            for e, f in Image.registered_extensions().items()
            if f in Image.OPEN
        ]
        #
        # paths = [
        #     Path(p)
        #     for p in glob(batch_input_glob, recursive=batch_input_recursive)
        #     if '.' + p.split('.').pop().lower() in supported_extensions
        # ]
        if target_paths:
            paths = [
                Path(p) for p in target_paths
                if '.' + p.split('.').pop().lower() in supported_extensions
            ]
        else:
            paths = [
                Path(p)
                for p in glob(batch_input_glob, recursive=True)
                if '.' + p.split('.').pop().lower() in supported_extensions
            ]

        print(f'found {len(paths)} image file(s)')
        # print(paths)

        # allow to truncate oversized image instead of raising exception ending the hole progress
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        paths_count = len(paths)
        for i, path in enumerate(paths):

            # If open image failed,  skip and print error msg instead of raising exception ending the hole progress
            try:
                image = Image.open(path)
            except UnidentifiedImageError:
                # just in case, user has mysterious file...
                print(f'${path} is not supported image type')
                continue
            except Exception as e:
                traceback.print_exc()
                continue
            # guess the output path
            base_dir_last = Path(base_dir).parts[-1]
            base_dir_last_idx = path.parts.index(base_dir_last)
            output_dir = Path(
                batch_output_dir) if batch_output_dir else Path(base_dir)
            output_dir = output_dir.joinpath(
                *path.parts[base_dir_last_idx + 1:]).parent

            output_dir.mkdir(0o777, True, True)

            # format output filename
            format_info = format.Info(path, 'txt')

            try:
                formatted_output_filename = format.pattern.sub(
                    lambda m: format.format(m, format_info),
                    batch_output_filename_format
                )
            except (TypeError, ValueError) as error:
                return ['', None, None, str(error)]

            output_path = output_dir.joinpath(
                formatted_output_filename
            )

            output = []

            if output_path.is_file():
                output.append(output_path.read_text(errors='ignore').strip())

                if batch_output_action_on_conflict == 'ignore':
                    print(f'skipping {path}')
                    continue

            ratings, tags = interrogator.interrogate(image)
            processed_tags = Interrogator.postprocess_tags(
                tags,
                *postprocess_opts
            )

            # TODO: switch for less print
            print(
                f'found {len(processed_tags)} tags out of {len(tags)} from {path} with {tf_device_name}'
            )

            if logger:
                logger.log(i/paths_count, f'{i}/{paths_count}')

            plain_tags = ', '.join(processed_tags)

            if batch_output_action_on_conflict == 'copy':
                output = [plain_tags]
            elif batch_output_action_on_conflict == 'prepend':
                output.insert(0, plain_tags)
            else:  # 'add'
                output.append(plain_tags)

            if batch_remove_duplicated_tag:
                output_path.write_text(
                    ', '.join(
                        OrderedDict.fromkeys(
                            map(str.strip, ','.join(output).split(','))
                        )
                    ),
                    encoding='utf-8'
                )
            else:
                output_path.write_text(
                    ', '.join(output),
                    encoding='utf-8'
                )

            if batch_output_save_json:
                output_path.with_suffix('.json').write_text(
                    json.dumps([ratings, tags])
                )

        print('all done :)')

    if unload_model_after_running:
        interrogator.unload()

    return ['', None, None, '']