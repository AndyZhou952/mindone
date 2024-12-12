import os
import logging
import mindspore as ms
from models.pllava import PllavaProcessor, PllavaForConditionalGeneration, PllavaConfig
from models.pipeline import TextGenerator

def ckpt_key_match(checkpoint_data):
    ckpt_prefix = "language_model.base_model.model.model"
    model_prefix = "language_model.model"

    aligned_checkpoint = {}
    for key, value in checkpoint_data.items():
        if key.startswith(ckpt_prefix):
            new_key = key.replace(ckpt_prefix, model_prefix, 1)
        else:
            new_key = key

        if "self_attn.q_proj.base_layer.weight" in new_key:
            new_key = new_key.replace("self_attn.q_proj.base_layer.weight", "self_attn.q_proj.weight")
        elif "self_attn.v_proj.base_layer.weight" in new_key:
            new_key = new_key.replace("self_attn.v_proj.base_layer.weight", "self_attn.v_proj.weight")
        if "self_attn.q_proj.lora_" in new_key or "self_attn.v_proj.lora_" in new_key:
            continue  # no need for lora

        if "lm_head.weight" in new_key:
            new_key = "language_model.lm_head.weight"

        aligned_checkpoint[new_key] = value

    return aligned_checkpoint

def load_pllava(repo_id, num_frames, pooling_shape=(16,12,12), vision_hidden_size = 1024, text_hidden_size = 4096):
    kwargs = {
        'num_frames': num_frames,
    }
    if num_frames == 0:
        kwargs.update(pooling_shape=(0,12,12))

    config = PllavaConfig.from_pretrained(
        repo_id,
        vision_hidden_size = vision_hidden_size,
        text_hidden_size = text_hidden_size,
        pooling_shape=pooling_shape,
        **kwargs,
    )

    model = PllavaForConditionalGeneration(config)
    model_path = os.path.join(repo_id, 'model.ckpt')
    logging.info(f"Loading model from {model_path}")

    checkpoint_data = ms.load_checkpoint(model_path)
    aligned_checkpoint = ckpt_key_match(checkpoint_data)

    ms.load_param_into_net(model, aligned_checkpoint, strict_load=True)

    try:
        processor = PllavaProcessor.from_pretrained(repo_id)
    except Exception:
        processor = PllavaProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')

    model.set_train(False)
    return model, processor

def pllava_answer(model, processor, img_list, prompt,
                  do_sample=False, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
                  repetition_penalty=1.0, length_penalty=1, temperature=1.0):
    inputs = processor(text=prompt, images=img_list, return_tensors="np")
    if inputs['pixel_values'] is None:
        inputs.pop('pixel_values')
    inputs = {k: ms.Tensor(v) for k, v in inputs.items()}

    model.set_train(False)
    pipeline = TextGenerator(model, max_new_tokens = max_new_tokens, use_kv_cache=True)
    output_token = pipeline.generate(**inputs, media_type='video',
                                  do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                  min_length=min_length, top_p=top_p, repetition_penalty=repetition_penalty,
                                  length_penalty=length_penalty, temperature=temperature)
    output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return output_token.asnumpy(), output_text.strip()


