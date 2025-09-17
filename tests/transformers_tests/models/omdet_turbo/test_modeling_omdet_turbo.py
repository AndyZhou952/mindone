import numpy as np
import pytest
import torch
from transformers import OmDetTurboConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-2, "fp16": 5e-2, "bf16": 5e-2}
MODES = [1]


class OmDetTurboModelTester:
    def __init__(
        self,
        batch_size=2,
        image_size=64,
        classes_per_image=2,
        classes_seq_len=6,
        tasks_seq_len=6,
        encoder_hidden_dim=64,
        decoder_hidden_dim=64,
        d_model=64,
        encoder_attention_heads=4,
        decoder_num_heads=4,
        encoder_layers=1,
        decoder_num_layers=2,
        encoder_dim_feedforward=128,
        decoder_dim_feedforward=256,
        num_queries=10,
        torch_dtype="float32",
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.classes_per_image = classes_per_image
        self.total_classes = batch_size * classes_per_image
        self.classes_seq_len = classes_seq_len
        self.tasks_seq_len = tasks_seq_len

        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_num_heads = decoder_num_heads
        self.encoder_layers = encoder_layers
        self.decoder_num_layers = decoder_num_layers
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.decoder_dim_feedforward = decoder_dim_feedforward
        self.num_queries = num_queries
        self.torch_dtype = torch_dtype

    def get_config(self):
        backbone_config = {
            "model_type": "swin",
            "window_size": 7,
            "image_size": self.image_size,
            "embed_dim": 32,
            "depths": [1, 1, 1, 1],
            "num_heads": [1, 2, 4, 8],
            "out_indices": [1, 2, 3],
        }

        text_config = {"model_type": "clip_text_model"}

        config = OmDetTurboConfig(
            text_config=text_config,
            use_timm_backbone=False,
            backbone=None,
            backbone_config=backbone_config,
            image_size=self.image_size,
            # Hybrid encoder sizes
            encoder_hidden_dim=self.encoder_hidden_dim,
            vision_features_channels=[self.encoder_hidden_dim] * 3,
            encoder_in_channels=[64, 128, 256],
            encoder_layers=self.encoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            encoder_dim_feedforward=self.encoder_dim_feedforward,
            # Decoder sizes
            d_model=self.d_model,
            decoder_hidden_dim=self.decoder_hidden_dim,
            decoder_num_heads=self.decoder_num_heads,
            decoder_num_layers=self.decoder_num_layers,
            decoder_dim_feedforward=self.decoder_dim_feedforward,
            num_queries=self.num_queries,
            decoder_dropout=0.0,
            encoder_dropout=0.0,
            encoder_feedforward_dropout=0.0,
            torch_dtype=self.torch_dtype,
        )
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()

        pixel_values = ids_numpy([self.batch_size, 3, self.image_size, self.image_size], vocab_size=256).astype(
            np.float32
        )
        pixel_values = pixel_values / 255.0

        classes_input_ids = ids_numpy([self.total_classes, self.classes_seq_len], vocab_size=200)
        classes_attention_mask = np.ones_like(classes_input_ids, dtype=np.int64)

        tasks_input_ids = ids_numpy([self.batch_size, self.tasks_seq_len], vocab_size=200)
        tasks_attention_mask = np.ones_like(tasks_input_ids, dtype=np.int64)

        classes_structure = np.array([self.classes_per_image] * self.batch_size, dtype=np.int64)

        return (
            config,
            pixel_values,
            classes_input_ids,
            classes_attention_mask,
            tasks_input_ids,
            tasks_attention_mask,
            classes_structure,
        )


# Prepare once
model_tester = OmDetTurboModelTester()
(
    config,
    pixel_values,
    classes_input_ids,
    classes_attention_mask,
    tasks_input_ids,
    tasks_attention_mask,
    classes_structure,
) = model_tester.prepare_config_and_inputs()

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

    ms_inputs_kwargs.update({"return_dict": False})

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
