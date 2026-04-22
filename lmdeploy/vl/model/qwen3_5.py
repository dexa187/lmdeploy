# Copyright (c) OpenMMLab. All rights reserved.
import torch
from transformers import AutoProcessor

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.qwen2 import Qwen2VLModel
from lmdeploy.vl.model.utils import disable_logging

from .qwen3 import Qwen3VLModel


def check_transformers():
    try:
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3_5Model(Qwen3VLModel):
    """Qwen3_5 model."""

    _arch = ['Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # image tokens
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id

        # video tokens
        self.video_token = self.processor.video_token
        self.video_token_id = self.processor.video_token_id

        # vision start and end tokens
        self.vision_start_token = self.processor.vision_start_token
        self.vision_end_token = self.processor.vision_end_token

    def build_model(self):
        """Load Hugging Face vision tower only (TurboMind runs language on TM)."""
        check_transformers()
        arch = self.hf_config.architectures[0]
        if arch == 'Qwen3_5ForConditionalGeneration':
            from transformers import Qwen3_5ForConditionalGeneration as AutoModelCls

            no_split_module_classes = ['Qwen3_5VisionBlock']
        elif arch == 'Qwen3_5MoeForConditionalGeneration':
            from transformers import Qwen3_5MoeForConditionalGeneration as AutoModelCls

            no_split_module_classes = ['Qwen3_5MoeVisionBlock']
        else:
            raise ValueError(f'Unsupported arch={arch}')

        from accelerate import init_empty_weights

        if self.with_llm:
            full = AutoModelCls.from_pretrained(self.model_path, device_map='cpu')
            self._hf_full_model = full
            wrap = type('Qwen35VisionWrapper', (), {})()
            wrap.visual = full.model.visual
            self.model = wrap
            return

        with init_empty_weights():
            config = self.hf_config
            if hasattr(config, 'text_config'):
                config.text_config.tie_word_embeddings = False
            if hasattr(config, 'tie_word_embeddings'):
                config.tie_word_embeddings = False
            model = AutoModelCls._from_config(config)
            model.visual = model.model.visual
            del model.model
            del model.lm_head

        target_dtype = torch.float16
        if hasattr(self.hf_config, 'torch_dtype') and self.hf_config.torch_dtype is not None:
            target_dtype = self.hf_config.torch_dtype
            if isinstance(target_dtype, str):
                target_dtype = getattr(torch, target_dtype.split('.')[-1])
        model = model.to(dtype=target_dtype)

        from accelerate import load_checkpoint_and_dispatch

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map='auto',
                max_memory=self.max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=target_dtype,
            )
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, messages: list[dict], max_batch_size: int = 1) -> list[dict]:
        """Extract image features for TurboMind (HF vision forward)."""
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        for item in inputs:
            if item.get('modality') == Modality.VIDEO:
                raise NotImplementedError(
                    'Qwen3.5 / Qwen3.6 video with TurboMind is not supported yet; use backend=pytorch.')

        dtype = next(self.model.visual.parameters()).dtype
        device = next(self.model.visual.parameters()).device
        outputs = []
        merge_sq = self.model.visual.spatial_merge_size**2
        for idx in range(0, len(inputs), max_batch_size):
            batch = inputs[idx:idx + max_batch_size]
            pixel_values = [x['pixel_values'].to(dtype=dtype) for x in batch]
            image_grid_thw = [x['image_grid_thw'] for x in batch]
            pixel_values = torch.cat(pixel_values, dim=0).to(device)
            image_grid_thw = torch.cat(image_grid_thw, dim=0).to(device)
            vision_out = self.model.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
            image_embeds = vision_out.pooler_output
            split_sizes = (image_grid_thw.prod(dim=-1) // merge_sq).tolist()
            image_embeds = image_embeds.split(split_sizes)
            outputs.extend(image_embeds)
        messages.append(dict(role='forward', content=outputs))
        return messages

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     chat_template_kwargs=None,
                     **kwargs):
        prompt, _ = self.proc_messages(messages, chat_template, sequence_start, chat_template_kwargs)
        info = VisionModel.to_turbomind_aux(self, messages, prompt, self.image_token, tokenizer, sequence_start)
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        grid_thws = [x['image_grid_thw'].tolist()[0] for x in inputs]
        seq_len = len(info['input_ids'])
        ranges = info['input_embedding_ranges']
        mrope_position_ids, mrope_position_delta = Qwen2VLModel.get_mrope_info(seq_len, grid_thws, ranges)
        meta = dict(mrope_position_ids=mrope_position_ids, mrope_position_delta=mrope_position_delta)
        info.update(dict(input_meta=meta))
        return info
