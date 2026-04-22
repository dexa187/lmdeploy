# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.archs import check_vl_llm


def test_qwen35_conditional_turbomind_is_vl():
    cfg = {'architectures': ['Qwen3_5ForConditionalGeneration']}
    assert check_vl_llm('turbomind', cfg) is True


def test_qwen35_moe_turbomind_is_vl():
    cfg = {'architectures': ['Qwen3_5MoeForConditionalGeneration']}
    assert check_vl_llm('turbomind', cfg) is True


def test_qwen35_text_only_not_vl():
    cfg = {'architectures': ['Qwen3_5ForCausalLM']}
    assert check_vl_llm('turbomind', cfg) is False
