import copy
import itertools
import re
import os
import json
from enum import auto, Enum
import dataclasses
from typing import Any, List

from PIL import Image
import cv2
import imageio
import numpy as np
import mindspore as ms
import mindnlp.core.ops as ops
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from moviepy.editor import VideoFileClip

from decord import VideoReader, cpu # This is Terrible, if you have this line of import in front of torch, will cause model.to(device) to hang
from mindnlp.transformers import StoppingCriteria, StoppingCriteriaList

from pipeline.captioning.pllava.utils.easydict import EasyDict
from pipeline.datasets.utils import read_video_av

IMAGE_TOKEN = "<image>"


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()

class MultiModalConvStyle(Enum):
    """Different separator style."""
    MM_ALONE = 'mm_alone'
    MM_INTERLEAF = 'mm_inferleaf'

def dump_json(obj_serializable ,save_dir_path, json_file_name):
    os.makedirs(save_dir_path, exist_ok=True)
    save_path = os.path.join(save_dir_path, json_file_name)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj_serializable, f, indent=4, ensure_ascii=False, )

def load_json(load_dir_path, json_file_name):
    
    load_path = os.path.join(load_dir_path, json_file_name)
    if not os.path.exists(load_path):
        return None
    with open(load_path, 'r', encoding='utf-8') as f:
        obj_serializable = json.load(f)
    return obj_serializable



@dataclasses.dataclass
class Conversation(EasyDict):
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    sep: List[str]
    mm_token: str
    
    mm_style: MultiModalConvStyle = MultiModalConvStyle.MM_INTERLEAF
    pre_query_prompt: str=None
    post_query_prompt: str=None
    answer_prompt: str=None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.sep, str):
            self.sep = [self.sep for _ in self.roles]

    def get_prompt(self):
        sep = [self.sep for _ in self.roles] if isinstance(self.sep, str) else self.sep  # if only one sep given, then both sep are the sames
        sep = dict(zip(self.roles, sep))
        ret = self.system + sep[self.roles[0]] if self.system != "" else ""
        for i, (role, message) in enumerate(self.messages):
            # if is last msg(the prompt for assistant), if answer prompt exists, no sep added
            if i+1 == len(self.messages):
                if role != self.roles[-1]: # last role is not the model
                    ret += role + message + sep[role] + self.roles[-1]
                else:
                    ret += role + message
            else:
                ret += role + message + sep[role]
        return ret
    # def get_prompt_multichoice(self):
    #     pass
    def user_query(self, query=None, pre_query_prompt=None, post_query_prompt=None, is_mm=False, num_mm_token=1):
        if post_query_prompt is not None:
            query = f"{query} {post_query_prompt}"

        if pre_query_prompt is not None:
            query = f"{pre_query_prompt} {query}"
        role = self.roles[0]
        # TODO: remove the num_mm_token and hack the self.mm_token outside
        if is_mm:
            mm_str = num_mm_token*self.mm_token[:-1] + self.mm_token[-1]
            if self.mm_style == MultiModalConvStyle.MM_ALONE:
                self._append_message(role, mm_str)
            elif self.mm_style == MultiModalConvStyle.MM_INTERLEAF:
                if self.mm_token not in query:
                    query = f'{mm_str} {query}'
        self._append_message(role, query)
    
    def assistant_response(self, response, pre_query_prompt=None, post_query_prompt=None):
        if post_query_prompt is not None:
            response = f"{response} {post_query_prompt}"

        if pre_query_prompt is not None:
            response = f"{post_query_prompt} {response}"

        role = self.roles[1]
        self._append_message(role, response)
    
    def _append_message(self, role, message):
        message = '' if message is None else message
        self.messages.append([role, message])

    def copy(self):
        return copy.deepcopy(self)

conv_video_chatgpt_v1 = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
)


conv_plain_v1 = Conversation(
    system="",
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token='<image>'
)

# Attention to the roles[0] "USER: " has a space!
conv_eval_vcg = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)

conv_eval_vcg_llavanext = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)

SYSTEM_MVBENCH="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
conv_eval_mvbench = Conversation(
    system=SYSTEM_MVBENCH,
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)
conv_eval_mvbench_llavanext = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)


conv_eval_videoqabench = Conversation(
    system="",
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
    pre_query_prompt="The input consists of a sequence of key frames from a video. Answer the question concisely first and followed by significant events, characters, or objects that appear throughout the frames. Question:",
    post_query_prompt="\n",
    answer_prompt='\nAnswer: In the video,'
)

conv_eval_videoqa_llavanext = Conversation(
    system="<|im_start|>system\nAnswer the question.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
    pre_query_prompt="The input consists of a sequence of key frames from a video. Answer the question concisely first and followed by significant events, characters, or objects that appear throughout the frames. Question:",
    post_query_prompt="\n",
    answer_prompt='\nAnswer: In the video,'
)


SYSTEM_RECAPTION="""You are a powerful Video Magic ChatBot, a large vision-language assistant. 
You are able to understand the video content that the user provides and assist the user in a video recaptioning task.
The user will provide you with the video and maybe some extra noisy information to help you out. Make use of the information in a proper way to be competent for the recaption job
### INSTRUCTIONS:
1. Follow the user's instruction.
2. Be critical yet believe in yourself.
"""
conv_eval_recaption = Conversation(
    system=SYSTEM_RECAPTION,
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)


conv_eval_recaption_llavanext = Conversation(
    system=SYSTEM_RECAPTION,
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)


conv_templates = {
    "plain": conv_plain_v1,
    "eval_vcgbench": conv_eval_vcg,
    "eval_vcg_llavanext": conv_eval_vcg_llavanext,
    "eval_mvbench": conv_eval_mvbench,
    "eval_mvbench_llavanext": conv_eval_mvbench_llavanext,
    "eval_videoqabench": conv_eval_videoqabench,
    "eval_videoqa_llavanext": conv_eval_videoqa_llavanext,
    "eval_recaption": conv_eval_recaption,
    "eval_recaption_llavanext": conv_eval_recaption_llavanext,
}


class EvalDataset():

    def __init__(self, num_segments, test_ratio=None):
        super().__init__()
        self.num_segments = num_segments
        self.test_ratio = test_ratio
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_clip_gif,
            'frame': self.read_frame,
        }
        
    def __getitem__(self, index) -> Any:
        raise NotImplementedError('')
        
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        return images_group
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
                if len(images_group) == len(frame_indices):
                    break

        # might be some really short videos in the gif datasets
        if len(images_group) < self.num_segments:
            multiplier = int(self.num_segments/len(images_group)) + 1
            images_group = [image for _ in range(multiplier) for image in images_group][:self.num_segments]
            assert len(images_group) == self.num_segments

        return images_group
    
    def read_clip_gif(self, video_path, bound=None, fps=25):
        gif = VideoFileClip(video_path)
        frames = gif.iter_frames()
        max_frame = gif.reader.nframes - 1
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(frames):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)

        # might be some really short videos in the gif datasets
        if len(images_group) < self.num_segments:
            multiplier = int(self.num_segments/len(images_group)) + 1
            images_group = [image for _ in range(multiplier) for image in images_group][:self.num_segments]
            assert len(images_group) == self.num_segments

        return images_group
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        return images_group

    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # self.data_list = self.data_list[::200] # debug
        if self.test_ratio is None:
            self.data_list = self.data_list[rank::world_size]
        else:
            np.random.RandomState(42).shuffle(self.data_list)
            if isinstance(self.test_ratio, float):
                num_samples = int(len(self.data_list) * self.test_ratio)
            else:
                num_samples = int(self.test_ratio)
            self.data_list = self.data_list[rank:num_samples:world_size]


class ChatPllava:
    print_res=True
    do_sample=False
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def ask(self, text, conv: Conversation, system):
        conv.system = system
        conv.user_query(text, )
        return conv

    def answer(self, conv: Conversation, img_list, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0):
        prompt = conv.get_prompt()
        if prompt.count(conv.mm_token) < len(img_list):
            diff_mm_num = len(img_list) - prompt.count(conv.mm_token)
            for i in range(diff_mm_num):
                conv.user_query("", is_mm=True)
            prompt = conv.get_prompt()
            
        inputs = self.processor(text=prompt, images=img_list, return_tensors="ms")
        if inputs['pixel_values'] is None:
            inputs.pop('pixel_values')

#        with mindnlp.core.no_grad():
        self.model.set_train(False)
        output_token = self.model.generate(**inputs, media_type='video',
                                        do_sample=self.do_sample,max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length, 
                                        top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature,
                                        ) # don't need to long for the choice.
        output_text = self.processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if self.print_res:
            print('###PROMPT: ', prompt)
            print('###LM OUTPUT TEXT', output_text)
        # <|im_start|> encode and then decode would extend a space at following, this is insane...
        if conv.roles[-1] == "<|im_start|>assistant\n":
            split_tag = "<|im_start|> assistant\n"
        else:
            split_tag = conv.roles[-1]
        output_text = output_text.split(split_tag)[-1].rstrip(conv.sep[1])
        conv.assistant_response(output_text)
        return output_text, output_token.asnumpy(), conv
    
        
    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(self, video_path, num_segments):
        try:
            vframes, info = read_video_av(
                video_path,
                pts_unit="sec",
                output_format="THWC"
            )
            total_num_frames = len(vframes)

            if total_num_frames == 0:
                raise ValueError("No frames found in the video.")

            frame_indices = self.get_index(total_num_frames, num_segments)
            images_group = list()
            for frame_index in frame_indices:
                images_group.append(vframes[frame_index])
            images_group = ms.Tensor(images_group, dtype=ms.uint8)

            return images_group

        except (IndexError, ValueError) as e:
            # TODO: temporary fix to skip videos, may need to fix the logic later
            # skip this video if error occurs
            print(f"Error processing video at {video_path}: {e}")
            return None


    def upload_video(self, image, conv: Conversation, img_list: list[list], num_segments=None):
        num_segments = self.model.config.num_frames if num_segments is None else num_segments 
        if isinstance(image, str):  # is an image path
            vid = self.load_video(image, num_segments=num_segments)
        else:
            raise NotImplementedError
        if vid is not None:
            img_list.append(vid)
            conv.user_query("", is_mm=True)
            msg = "Received."
            # self.conv.append_message(self.conv.roles[1], msg)
            return msg, img_list, conv
        else:
            return None, None, None # error handling

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: ms.Tensor, scores: ms.Tensor):
        for stop in self.stops:
            if ops.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
