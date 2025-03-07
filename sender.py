import logging
import time
from collections import defaultdict
from datetime import datetime

import subprocess

import numpy as np
import torch

from model import Upscaler


def initialize_streams(endpoint, fps, video_size):
    in_cmd = f'ffmpeg -f x11grab -r {fps} -s {video_size} -i :0.0 -f rawvideo -pix_fmt rgb24 -'
    ffmpeg_input_stream = subprocess.Popen(
        in_cmd.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Initialize FFmpeg subprocess
    out_cmd = f'ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size {video_size} -framerate {fps} -i - -c:v libx264 -preset fast -f mpegts {endpoint}'
    ffmpeg_output_stream = subprocess.Popen(
        out_cmd.split(" "),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return ffmpeg_input_stream, ffmpeg_output_stream


def ffmpeg_close(ff_proc):
    # Close the input stream and wait for FFmpeg to finish
    if ff_proc.stdin is not None:
        ff_proc.stdin.close()
    if ff_proc.stdout is not None:
        ff_proc.stdout.close()
    ff_proc.wait()
    # Print any remaining errors from FFmpeg
    if ff_proc.stderr.readline():
        print("Output stream errors:")
        while True:
            line = ff_proc.stderr.readline()
            if not line:
                break
            print(line.decode('utf-8').strip())


def readable_numbers(num: int):
    s = 1000
    postfix = {
        'b': 1,
        'Kb': s,
        'Mb': s ** 2,
        'Gb': s ** 3,
        'Tb': s ** 4,
        'Pb': s ** 5,
    }
    for pf, val in postfix.items():
        if num / val < s:
            return f'{round(num / val)}{pf}'
    return f'{round(num/postfix["p"])}p'

def main():
    # Initialize mss
    fps = 30
    width = 800
    height = 600
    video_size = f'{width}x{height}'
    endpoint = 'udp://@localhost:2222'

    ffmpeg_input_stream, ffmpeg_output_stream = initialize_streams(endpoint, fps, video_size)

    logging_interval = 100
    frame_idx = 0
    log_data = defaultdict(int)
    try:
        wait_time = 1 / fps
        logging.info(f'Streaming to {endpoint}')
        while True:
            frame_idx += 1
            start_time = datetime.now()
            raw_frame: bytes = ffmpeg_input_stream.stdout.read(width * height * 3)
            if not raw_frame:
                break
            log_data['data_sent'] += len(raw_frame)
            frame = process_frame(raw_frame, height, width)

            ffmpeg_output_stream.stdin.write(frame.tobytes())

            proc_elapsed = (datetime.now() - start_time).total_seconds()
            if proc_elapsed < wait_time:
                time.sleep(wait_time - proc_elapsed)
            else:
                log_data['delayed_frames'] += 1

            if frame_idx % logging_interval == 0:
                logging.info(', '.join([f'{k}={readable_numbers(v)}' for k, v in log_data.items()]))
                log_data.clear()
    finally:
        ffmpeg_close(ffmpeg_input_stream)
        ffmpeg_close(ffmpeg_output_stream)


model = Upscaler(upscale_factor=2)
def process_frame(raw_frame: bytes, height, width) -> np.array:
    # Convert to NumPy array
    frame = torch.from_numpy(np.frombuffer(raw_frame, dtype=np.uint8).copy()).reshape((1, height, width, 3)).type(torch.float16)
    predict_frame = model.update(frame / 255.0)
    predict_frame = predict_frame.squeeze() * 255.0
    return predict_frame.type(torch.uint8).numpy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
