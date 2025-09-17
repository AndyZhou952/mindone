import numpy as np
import pytest
import torch
from transformers import OmDetTurboConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 1e-2, "fp16": 1e-2, "bf16": 1e-2}
MODES = [1]


class OmDetTurboModelTester:
    def __init__(
        self,
        batch_size=1,
        # vision
        image_size=56,
        num_channels=3,
        class_seq_len=5,
        task_seq_len=7,
        classes_per_sample=2,
        encoder_hidden_dim=64,
        decoder_hidden_dim=64,
        num_queries=16,
        encoder_layers=1,
        decoder_layers=2,
        torch_dtype="float32",
        use_timm_backbone=False,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels

        self.class_seq_len = class_seq_len
        self.task_seq_len = task_seq_len
        self.classes_per_sample = classes_per_sample

        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_queries = num_queries
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.torch_dtype = torch_dtype
        self.use_timm_backbone = use_timm_backbone

        self.encoder_in_channels = [192, 384, 768]
        self.vision_features_channels = [256, 256, 256]

    def get_config(self):
        config = OmDetTurboConfig(
            use_timm_backbone=self.use_timm_backbone,
            image_size=self.image_size,
            encoder_hidden_dim=self.encoder_hidden_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            num_queries=self.num_queries,
            encoder_layers=self.encoder_layers,
            decoder_num_layers=self.decoder_layers,
            encoder_in_channels=self.encoder_in_channels,
            vision_features_channels=self.vision_features_channels,
            encoder_attention_heads=4,
            decoder_num_heads=4,
            encoder_dim_feedforward=4 * self.encoder_hidden_dim,
            decoder_dim_feedforward=4 * self.decoder_hidden_dim,
            text_config={"model_type": "clip_text_model"},
            torch_dtype=self.torch_dtype,
            layer_norm_eps=1e-5,
            batch_norm_eps=1e-5,
        )
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()

        B = self.batch_size
        H = W = self.image_size
        C = self.num_channels

        pixel_values = ids_numpy([B, C, H, W], vocab_size=256).astype(np.float32)
        pixel_values = (pixel_values / 255.0) * 2.0 - 1.0

        total_classes = B * self.classes_per_sample
        classes_input_ids = ids_numpy([total_classes, self.class_seq_len], vocab_size=50)
        classes_attention_mask = (ids_numpy([total_classes, self.class_seq_len], vocab_size=2) > 0).astype(np.int64)

        tasks_input_ids = ids_numpy([B, self.task_seq_len], vocab_size=80)
        tasks_attention_mask = (ids_numpy([B, self.task_seq_len], vocab_size=2) > 0).astype(np.int64)
        classes_structure = np.array([self.classes_per_sample] * B, dtype=np.int64)

        return (
            config,
            pixel_values,
            classes_input_ids,
            classes_attention_mask,
            tasks_input_ids,
            tasks_attention_mask,
            classes_structure,
        )


tester = OmDetTurboModelTester()
(
    config,
    pixel_values,
    classes_input_ids,
    classes_attention_mask,
    tasks_input_ids,
    tasks_attention_mask,
    classes_structure,
) = tester.prepare_config_and_inputs()


TEST_CASES = [
    [
        "OmDetTurboForObjectDetection",
        "transformers.OmDetTurboForObjectDetection",
        "mindone.transformers.OmDetTurboForObjectDetection",
        (config,),
        {},
        (),
        {
            "pixel_values": pixel_values,
            "classes_input_ids": classes_input_ids,
            "classes_attention_mask": classes_attention_mask,
            "tasks_input_ids": tasks_input_ids,
            "tasks_attention_mask": tasks_attention_mask,
            "classes_structure": classes_structure,
            "return_dict": True,
        },
        {
            "decoder_coord_logits": 1,
            "decoder_class_logits": 2,
            "encoder_coord_logits": 5,
            "encoder_class_logits": 6,
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in TEST_CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)

    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    pt_inputs_kwargs["return_dict"] = True
    ms_inputs_kwargs["return_dict"] = False

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
