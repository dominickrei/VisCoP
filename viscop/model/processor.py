# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for ViSCoP.
"""
import copy
import math
import warnings
from typing import List, Union, Dict, Optional

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from viscop.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_PROBE_TOKEN, IGNORE_INDEX
from viscop.mm_utils import load_video, load_images


DEFAULT_CHAT_TEMPLATE = """
{%- set identifier = 'im' %}
{% for message in messages %}
    {% if message['role'] == 'stream' %}
        {% set identifier = 'stream' %}
    {% else %}
        {% set identifier = 'im' %}
    {% endif %}
    {{- '<|' + identifier + '_start|>' + message['role'] + '\n' -}}
    {% if message['content'] is string %}
        {{- message['content'] + '<|' + identifier + '_end|>\n' -}}
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% if 'time' in content %}
                    {{- 'Time ' + content['time'] | round(1) | string + 's: ' -}}
                {% endif %}
"""
DEFAULT_CHAT_TEMPLATE += """
                {{- '%s\n' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
            {% elif content['type'] == 'video' or 'video' in content or 'video_url' in content %}
                {% for i in range(content['num_frames']) %}
                    {% if 'timestamps' in content %}
                        {{- 'Time ' + content['timestamps'][i] | round(1) | string + 's:' -}}
                    {% endif %}
                    {% if i < content['num_frames'] - 1 %}
"""
DEFAULT_CHAT_TEMPLATE += """
                        {{- '%s,' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
                    {% else %}
"""
DEFAULT_CHAT_TEMPLATE += """
                        {{- '%s\n' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
                    {% endif %}
                {% endfor %}
            {% elif content['type'] == 'text' or 'text' in content %}
                {{- content['text'] -}}
            {% endif %}
        {% endfor %}
        {{- '<|' + identifier + '_end|>\n' -}}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' -}}
{% endif %}
"""

# Types used for conversation processing (specifically in gradio interface)
from viscop.model.viscop_vision_encoder.image_processing_viscop import is_valid_video, is_valid_image

def is_named_image(image) -> bool:
    return isinstance(image, (list, tuple)) and \
        len(image) == 2 and \
        isinstance(image[0], str) and \
        image[0] in ["image", "video"] and \
        (is_valid_image(image[1]) or is_valid_video(image[1]))

def make_batched_images(images) -> List[List[ImageInput]]:
    if isinstance(images, (list, tuple)) and all(is_named_image(image) for image in images):
        # list of named images
        return [image[0] for image in images], [image[1] for image in images]
    elif isinstance(images, (list, tuple)) and all(is_valid_image(image) or is_valid_video(image) for image in images):
        # list of images/videos
        batch = []
        for image in images:
            if is_valid_video(image):
                batch.append(("video", image))
            elif is_valid_image(image):
                batch.append(("image", image))
            else:
                raise ValueError(f"Could not make batched images from {images}")
        return [x[0] for x in batch], [x[1] for x in batch]
    elif is_named_image(images):
        # named images
        return [images[0]], [image[1]]
    elif is_valid_video(images):
        # single video
        return ["video"], [images]
    elif is_valid_image(images):
        # single image
        return ["image"], [images]

    raise ValueError(f"Could not make batched images from {images}")

from typing import Any, Tuple, TypedDict
from collections import defaultdict
from PIL import Image
import numpy as np
import json

Conversation = List[Dict[str, Any]]
SingleImage = Union[Image.Image, np.ndarray, torch.Tensor]
SingleVideo = Union[List[SingleImage], np.ndarray, torch.Tensor]
BatchedImage = List[Union[SingleImage, SingleVideo]]
BatchedNamedImage = List[Tuple[str, Union[SingleImage, SingleVideo]]]


class ViSCoP_ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class ViSCoP_Processor(ProcessorMixin):
    r"""
    Modified from Qwen2VLProcessor
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, include_visual_tokens=True, include_visual_probes=False, num_visual_probes=32, **kwargs):
        if chat_template is None:
            chat_template = DEFAULT_CHAT_TEMPLATE
        # super().__init__(image_processor, tokenizer, chat_template=chat_template)
        tokenizer.chat_template = chat_template
        self.chat_template = chat_template
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.include_visual_tokens = include_visual_tokens
        self.include_visual_probes = include_visual_probes
        self.num_visual_probes = num_visual_probes

        self.generation_prompt = self._infer_generation_prompt()
        self.generation_prompt_ids = self.tokenizer.encode(self.generation_prompt, return_tensors="pt")
        self.generation_prompt_length = len(self.generation_prompt_ids[0])
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_generation_prompt(self):
        return self.generation_prompt

    def get_generation_prompt_ids(self):
        return self.generation_prompt_ids

    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    def load_images(self, *args, **kwargs):
        return load_images(*args, **kwargs)

    def _infer_generation_prompt(self):
        pseudo_message = [{"role": "user", "content": ""}]
        instruction = self.tokenizer.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=True)
        conversation = self.tokenizer.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=False)
        return instruction.replace(conversation, "")

    def _process_text_with_label(
        self,
        text: List[Dict],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        assert kwargs.pop("return_tensors", "pt") == "pt", "Only PyTorch tensors are supported when return_labels=True."
        assert isinstance(text[0], dict), "When return_labels=True, text must be a list of messages."

        input_ids_list = []
        targets_list = []
        sample_types_list = []
        image_idx = 0

        for message_idx, message in enumerate(text):
            # 1. set chat template and append image tokens
            if not self.include_visual_tokens:
                if isinstance(message['content'], list) and 'timestamps' in message['content'][0]:
                    del message['content'][0]['timestamps']

            prompt = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)

            if not self.include_visual_tokens:
                prompt = prompt.replace('<image>,', '')

            if self.include_visual_probes:
                if self.include_visual_tokens:
                    prompt = prompt.replace('<image>\n', '<image>\n' + DEFAULT_PROBE_TOKEN * self.num_visual_probes + '\n')
                else:
                    prompt = prompt.replace('<image>\n', DEFAULT_PROBE_TOKEN * self.num_visual_probes + '\n')

            prompt_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
            prompt = []
            for chunk_idx in range(len(prompt_chunks) - 1):
                prompt.append(prompt_chunks[chunk_idx])
                thw = grid_sizes[image_idx]
                prompt.append(DEFAULT_IMAGE_TOKEN * thw.prod().long())
                image_idx += 1
            prompt.append(prompt_chunks[-1])
            prompt = "".join(prompt)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            input_ids_list.append(input_ids)

            targets = torch.full_like(input_ids, IGNORE_INDEX)
            sample_types = torch.full_like(input_ids, IGNORE_INDEX)
            if message["role"] == "assistant":
                targets[self.generation_prompt_length:-1] = input_ids[self.generation_prompt_length:-1].clone()
            elif message["role"] == "stream":
                diff = torch.diff((input_ids == self.image_token_id).float())
                image_end_indices = torch.nonzero(diff < 0)[:, 0]
                targets[image_end_indices + 1] = input_ids[image_end_indices + 1]
                sample_types = targets.clone()
                sample_types[torch.logical_and(sample_types > 0, sample_types != self.eos_token_id)] = 0
                targets[-2] = input_ids[-2]    # <|im_end|>

            # if message_idx > 0 and text[message_idx - 1]["role"] == "stream":
            #     targets[0] = input_ids[0]
            #     # TODO: consider non-special tokens
            #     sample_types[0] = input_ids[0]

            targets_list.append(targets)
            sample_types_list.append(sample_types)

        # assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        targets = torch.cat(targets_list)
        sample_types = torch.cat(sample_types_list)
        types, counts = torch.unique(sample_types[sample_types > -1], return_counts=True)

        if len(types) > 0:
            target_num_samples = counts.amin()

            for type_id, type_count in zip(types, counts):
                if type_count > target_num_samples:
                    indices = torch.nonzero(sample_types == type_id)[:, 0]
                    random_selector = torch.randperm(indices.size(0))[:-target_num_samples]
                    targets[indices[random_selector]] = IGNORE_INDEX
                    sample_types[indices[random_selector]] = -1

        text_inputs = {
            "input_ids": torch.cat(input_ids_list),
            "labels": targets,
        }

        return text_inputs

    def _process_text_without_label(
        self,
        text: Union[List[str], List[Dict]],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        if isinstance(text[0], dict):
            if not self.include_visual_tokens:
                if isinstance(text[0]['content'], list) and 'timestamps' in text[0]['content'][0]:
                    del text[0]['content'][0]['timestamps']

            warnings.warn("Input text is a list of messages. Automatically convert it to a string with 'apply_chat_template' with generation prompt.")
            text = [self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)]

        if not self.include_visual_tokens:
            text[0] = text[0].replace('<image>,', '')

        if self.include_visual_probes:
            if self.include_visual_tokens:
                text[0] = text[0].replace('<image>\n', '<image>\n' + DEFAULT_PROBE_TOKEN * self.num_visual_probes + '\n')
            else:
                text[0] = text[0].replace('<image>\n', DEFAULT_PROBE_TOKEN * self.num_visual_probes + '\n')

        image_idx = 0
        for i in range(len(text)):
            while DEFAULT_IMAGE_TOKEN in text[i]:
                thw = grid_sizes[image_idx]
                text[i] = text[i].replace(DEFAULT_IMAGE_TOKEN, "<placeholder>" * thw.prod().long(), 1)
                image_idx += 1
            text[i] = text[i].replace("<placeholder>", DEFAULT_IMAGE_TOKEN)
        # assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."
        

        text_inputs = self.tokenizer(text, **kwargs)
        return text_inputs

    def process_text(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], List[Dict]],
        image_inputs: Dict[str, torch.Tensor] = {},
        return_labels: bool = False,
        **kwargs,
    ):
        kwargs.pop("padding", None)
        kwargs.pop("padding_side", None)

        if not isinstance(text, (list, tuple)):
            text = [text]
        assert len(text), "At least one text must be provided."

        grid_sizes = []
        for grid_size, merge_size in zip(image_inputs.get("grid_sizes", []), image_inputs.get("merge_sizes", [])):
            if not torch.all(grid_size[1:] % merge_size == 0):
                warnings.warn(f"Grid size {grid_size} is not divisible by merge size. Some undesired errors may occur.")
            if grid_size[0] == 1:
                grid_sizes.append(grid_size[1:] / merge_size)
            elif grid_size[0] > 1:
                grid_sizes.extend([grid_size[1:] / merge_size] * grid_size[0])

        if return_labels:
            return self._process_text_with_label(text, grid_sizes, **kwargs)
        return self._process_text_without_label(text, grid_sizes, **kwargs)

    def process_images(
        self,
        images: ImageInput = None,
        merge_size: Optional[int] = 1,
        **kwargs,
    ):
        if images is None:
            return {}
        image_inputs = self.image_processor(images=images, merge_size=merge_size, **kwargs)
        return image_inputs

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], List[Dict]] = None,
        images: ImageInput = None,
        merge_size: Optional[int] = 1,
        return_labels: bool = False,
        **kwargs: Unpack[ViSCoP_ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **grid_sizes** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            ViSCoP_ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        output_kwargs["text_kwargs"].pop("padding")
        output_kwargs["text_kwargs"].pop("padding_side")

        image_inputs = self.process_images(images, merge_size, **output_kwargs["images_kwargs"])
        text_inputs = self.process_text(text, image_inputs, return_labels, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    
    ### > Below we'll try to add the functions that allow the processor to be used for the gradio interface (which requires conversation processing)
    def gradio_call(
        self,
        conversation: Optional[Conversation] = None,
        images: Optional[Union[BatchedImage, BatchedNamedImage]] = None,
        return_labels: bool = False,
        **kwargs: Unpack[ViSCoP_ProcessorKwargs],
    ) -> BatchFeature:
        assert conversation is not None, "For gradio interface, conversation must be provided."
        
        return self._process_conversation(conversation, images, return_labels, **kwargs)

    def _process_conversation(
        self,
        conversation: Conversation,
        images: Optional[Union[BatchedImage, BatchedNamedImage]] = None,
        return_labels: bool = False,
        **kwargs: Unpack[ViSCoP_ProcessorKwargs],
    ) -> BatchFeature:
        assert isinstance(conversation, list), "Conversation must be a list of messages."

        if images is None:
            conversation = self._load_multimodal_data(conversation)
            images = self._gather_multimodal_data(conversation)

        output_kwargs = self._merge_kwargs(
            ViSCoP_ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.process_images_conv(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if return_labels:
            raise NotImplementedError("Processing conversation with labels is not implemented yet.")
            text_inputs = self._process_conversation_with_label(conversation, image_inputs, **kwargs)
        else:
            text_inputs = self._process_conversation_without_label(conversation, image_inputs, **kwargs)

        return BatchFeature(data={**text_inputs, **image_inputs})

    def process_images_conv(self, images: Union[BatchedImage, BatchedNamedImage], **kwargs):
        modals, images = make_batched_images(images)
        if not "merge_size" in kwargs:
            kwargs["merge_size"] = [
                1 if modal == "image" else 2
                for modal in modals
            ]

        # pass kwargs without add_system_prompt and add_generation_prompt to image processor
        kwargs = {k: v for k, v in kwargs.items() if k not in ["add_system_prompt", "add_generation_prompt"]}
        image_inputs = self.image_processor(images=images, **kwargs)
        image_inputs["modals"] = modals
        return image_inputs
        
    def _load_multimodal_data(self, conversation: Conversation):
        multimodal_info = defaultdict(list)
        new_conversation = []
        for message in conversation:
            new_message = {"role": message["role"]}
            if not isinstance(message["content"], (list, tuple)):
                new_message["content"] = message["content"]
                new_conversation.append(new_message)
                continue

            new_contents = []
            for content in message["content"]:
                if not isinstance(content, dict):
                    new_contents.append(content)
                    continue
                assert "type" in content, "Content must have 'type' field."
                if content["type"] in ["image", "video"] and content["type"] in content and isinstance(content[content["type"]], dict):
                    # TODO: support other types which are not compatible with json
                    load_args = content[content["type"]]
                    data_id = json.dumps({k: v for k, v in load_args.items() if not k in ["start_time", "end_time"]})
                    new_content = copy.deepcopy(content)
                    multimodal_info[data_id].append(new_content)
                    new_contents.append(new_content)
                else:
                    new_contents.append(content)

            new_message["content"] = new_contents
            new_conversation.append(new_message)

        for data_id, contents in multimodal_info.items():
            data_type = contents[0]["type"]
            if data_type == "image":
                image = self.load_images(contents[0][data_type]["image_path"])[0]
                for content in contents:
                    content["image"] = [image.copy()]

            elif data_type == "video":
                # TODO: start_time is None?
                start_times = [content["video"].get("start_time", 0.) for content in contents]
                end_times = [content["video"].get("end_time", float("inf")) for content in contents]

                load_args = contents[0][data_type]
                start_time, end_time = min(start_times), max(end_times)
                if start_time > 0:
                    load_args["start_time"] = start_time
                if end_time < float("inf"):
                    load_args["end_time"] = end_time
                images, timestamps = self.load_video(**load_args)

                for content, start_time, end_time in zip(contents, start_times, end_times):
                    cur_images, cur_timestamps = [], []
                    for image, timestamp in zip(images, timestamps):
                        if start_time <= timestamp <= end_time:
                            cur_images.append(image.copy())
                            cur_timestamps.append(timestamp)

                    content[data_type] = cur_images
                    content["num_frames"] = len(cur_images)
                    content["timestamps"] = cur_timestamps

        return new_conversation

    def _gather_multimodal_data(self, conversation: Conversation):
        images = []
        for message in conversation:
            if not isinstance(message["content"], (list, tuple)):
                continue
            for content in message["content"]:
                if not isinstance(content, dict):
                    continue
                if content["type"] == "video":
                    video = content["video"]
                    assert is_valid_video(video), f"Invalid video data: {video}."
                    images.append(("video", video))
                if content["type"] == "image":
                    image = content["image"]
                    images.append(("image", image))
        images = images if len(images) > 0 else None
        return images

    def _get_downsampled_grid_sizes(self, image_inputs: Dict[str, Any]):
        grid_sizes = []
        for grid_size, merge_size in zip(image_inputs.get("grid_sizes", []), image_inputs.get("merge_sizes", [])):
            if not torch.all(grid_size[1:] % merge_size == 0):
                warnings.warn(f"Grid size {grid_size} is not divisible by merge size. Some undesired errors may occur.")
            if grid_size[0] == 1:
                grid_sizes.append(grid_size[1:] / merge_size)
            elif grid_size[0] > 1:
                grid_sizes.extend([grid_size[1:] / merge_size] * grid_size[0])
        return grid_sizes

    def _get_visual_seq_len(self, grid_size: torch.Tensor):
        num_tokens = int(grid_size.prod().item())
        return num_tokens

    def process_text_conv(
        self,
        text: TextInput,
        image_inputs: Dict[str, Any],
        **kwargs,
    ):
        grid_sizes = self._get_downsampled_grid_sizes(image_inputs)

        kwargs.pop("padding", None)
        kwargs.pop("padding_side", None)

        if not self.include_visual_tokens:
            text = text.replace('<image>,', '')

        if self.include_visual_probes:
            if self.include_visual_tokens:
                text = text.replace('<image>\n', '<image>\n' + DEFAULT_PROBE_TOKEN * self.num_visual_probes + '\n')
            else:
                text = text.replace('<image>\n', DEFAULT_PROBE_TOKEN * self.num_visual_probes + '\n')

        image_idx = 0
        while DEFAULT_IMAGE_TOKEN in text:
            num_tokens = self._get_visual_seq_len(grid_sizes[image_idx])
            text = text.replace(DEFAULT_IMAGE_TOKEN, "<placeholder>" * num_tokens, 1)
            image_idx += 1
        text = text.replace("<placeholder>", DEFAULT_IMAGE_TOKEN)
 
        assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        text_inputs = self.tokenizer(text, **kwargs)
        return text_inputs

    def apply_chat_template(
        self,
        conversation: Conversation,
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        add_system_prompt: bool = False,
        add_generation_prompt: bool = False,
        image_token: Optional[str] = DEFAULT_IMAGE_TOKEN,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
            tokenize (`bool`, *optional*, defaults to `False`):
                Whether to tokenize the output or not.
            add_system_prompt (`bool`, *optional*, defaults to `False`):
                Whether to add the system prompt to the output or not.
            add_generation_prompt (`bool`, *optional*, defaults to `False`):
                Whether to add the generation prompt to the output or not.
            image_token (`Optional[str]`, *optional*, defaults to `<image>`):
                The token to use for indicating images in the conversation.
            **kwargs:
                Additional keyword arguments
        """

        if chat_template is None:
            # if self.chat_template is not None:
            #     chat_template = self.chat_template
            # else:
            #     raise ValueError(
            #         "No chat template is set for this processor. Please either set the `chat_template` attribute, "
            #         "or provide a chat template as an argument. See "
            #         "https://huggingface.co/docs/transformers/main/en/chat_templating for more information."
            #     )
            chat_template = "\n{%- set identifier = 'im' %}\n{% for message in messages %}\n    {% if add_system_prompt and loop.first and message['role'] != 'system' %}\n        {{- '<|im_start|>system\nYou are VideoLLaMA3 created by Alibaba DAMO Academy, a helpful assistant to help people understand images and videos.<|im_end|>\n' -}}\n    {% endif %}\n    {% if message['role'] == 'stream' %}\n        {% set identifier = 'stream' %}\n    {% else %}\n        {% set identifier = 'im' %}\n    {% endif %}\n    {{- '<|' + identifier + '_start|>' + message['role'] + '\n' -}}\n    {% if message['content'] is string %}\n        {{- message['content'] + '<|' + identifier + '_end|>\n' -}}\n    {% else %}\n        {% for content in message['content'] %}\n            {% if content is string %}\n                {{- content -}}\n            {% elif content['type'] == 'text' or 'text' in content %}\n                {{- content['text'] -}}\n            {% elif content['type'] == 'image' or 'image' in content %}\n                {% if 'timestamp' in content %}\n                    {{- 'Time ' + content['timestamp'] | round(1) | string + 's: ' -}}\n                {% endif %}\n                {{- image_token + '\n' -}}\n            {% elif content['type'] == 'video' or 'video' in content %}\n                {% for i in range(content['num_frames']) %}\n                    {% if 'timestamps' in content %}\n                        {{- 'Time ' + content['timestamps'][i] | round(1) | string + 's:' -}}\n                    {% endif %}\n                    {% if i < content['num_frames'] - 1 %}\n                        {{- image_token + ',' -}}\n                    {% else %}\n                        {{- image_token + '\n' -}}\n                    {% endif %}\n                {% endfor %}\n            {% endif %}\n        {% endfor %}\n        {% if identifier == 'stream' %}\n            {{- '<|' + identifier + '_end|>' -}}\n        {% else %}\n            {{- '<|' + identifier + '_end|>\n' -}}\n        {% endif %}\n    {% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n    {{- '<|im_start|>assistant\n' -}}\n{% endif %}\n"

        return self.tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=tokenize,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
            image_token=image_token,
            **kwargs
        )

    def _process_conversation_without_label(
        self,
        conversation: Conversation,
        image_inputs: Dict[str, Any],
        **kwargs,
    ):
        output_kwargs = self._merge_kwargs(
            ViSCoP_ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        chat_template_kwargs = {
            "chat_template": None,
            "add_system_prompt": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        prompt = self.apply_chat_template(
            conversation,
            tokenize=False,
            **chat_template_kwargs,
        )

        return self.process_text_conv(prompt, image_inputs, **output_kwargs["text_kwargs"])