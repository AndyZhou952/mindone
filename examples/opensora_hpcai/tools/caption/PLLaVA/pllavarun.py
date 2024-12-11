# pllavarun.py
from argparse import ArgumentParser
import mindspore as ms
import time
from eval_utils import load_video
from model_utils import load_pllava, pllava_answer

ms.set_context(pynative_synchronize=True, jit_config=dict(jit_level="O1"))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='./models/pllava-7b')
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file")
    parser.add_argument("--question", type=str, default="What is shown in this video?")
    parser.add_argument("--benchmark", action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()

    print('Initializing PLLaVA model...')
    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        args.num_frames,
        use_lora=args.use_lora,
        weight_dir=args.weight_dir,
        lora_alpha=args.lora_alpha
    )

    frames = load_video(args.video, args.num_frames)  # returns a list of PIL images
    prompt = "<video>\n" + args.question

    output_token, output_text = pllava_answer(
        model, processor, [frames], prompt,
        do_sample=False, max_new_tokens=200, num_beams=1, min_length=1,
        top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=1.0
    )

    if args.benchmark:
        # run again for benchmark
        start_time = time.time()
        output_text = pllava_answer(
            model, processor, [frames], prompt,
            do_sample=False, max_new_tokens=200, num_beams=1, min_length=1,
            top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=1.0
        )
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Tokens length: {output_token.shape[1]}")
        print(f"Time elapsed: {time_elapsed:.4f}")
        print(f'tokens per second: {(output_token.shape[1] / time_elapsed):.4f}')

    print(f"Response: {output_text}")

if __name__ == "__main__":
    main()