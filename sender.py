import logging
import threading
import time
from collections import defaultdict
from datetime import datetime

import subprocess

import numpy as np
import torch
from torch.autograd import set_detect_anomaly

from model import Upscaler


def initialize_streams(endpoint, fps, video_size):
    in_cmd = f'ffmpeg -f x11grab -video_size 1920x1080 -r {fps} -i :0 -f rawvideo -pix_fmt rgb24 -vf scale=800:600 -'
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


def readable_numbers(num):
    if type(num) == int:
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
        return f'{round(num / postfix["Pb"])}p'
    return num


logging_interval = 100
log_data = defaultdict(int)


class Trainer:
    def __init__(self):
        self.model = Upscaler(upscale_factor=2)
        self.frame_buffer = []

        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()
        self.state_dict_checkpoint = self.model.state_dict().copy()

    def loop(self):
        set_detect_anomaly(True)
        while True:
            if len(self.frame_buffer) == 0:
                continue

            with self.lock:
                frame = self.frame_buffer[0].detach().clone()

            frame = frame / 128 - 1
            predict_frame = self.model.update(frame)
            loss = self.model.backward(frame, predict_frame)
            log_data['loss'] += loss.detach().item() / logging_interval

            self.state_dict_checkpoint = self.model.state_dict().copy()

    def add(self, frame):
        with self.lock:
            self.frame_buffer.append(frame)

            if len(self.frame_buffer) >= 2:
                self.frame_buffer = self.frame_buffer[-2:]


def main():
    # Initialize mss
    fps = 30
    width = 800
    height = 600
    video_size = f'{width}x{height}'
    endpoint = 'udp://@localhost:2222'

    ffmpeg_input_stream, ffmpeg_output_stream = initialize_streams(endpoint, fps, video_size)
    frame_idx = 0
    trainer = Trainer()

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
            np_buffer = np.frombuffer(raw_frame, dtype=np.uint8).copy().reshape(height, width, 3)

            frame = process_frame(trainer, np_buffer, height, width)
            ffmpeg_output_stream.stdin.write(frame.tobytes())

            proc_elapsed = (datetime.now() - start_time).total_seconds()
            if proc_elapsed < wait_time:
                time.sleep(wait_time - proc_elapsed)
            else:
                log_data['delayed_frames'] += 1

            if frame_idx % logging_interval == 0:
                logging.info(', '.join([f'{k}={readable_numbers(v)}' for k, v in log_data.items()]))
                log_data.clear()

                model.load_state_dict(trainer.state_dict_checkpoint)
    finally:
        ffmpeg_close(ffmpeg_input_stream)
        ffmpeg_close(ffmpeg_output_stream)


model = Upscaler(upscale_factor=2)


def process_frame(trainer, np_frame: np.array, height, width) -> np.array:
    # Convert np to tensor
    frame = torch.from_numpy(np_frame).reshape((1, height, width, 3)).type(model.dtype)
    with torch.no_grad():
        trainer.add(frame.detach().clone())
        predict_frame = model.update(frame / 128 - 1)

    predict_frame = (predict_frame.squeeze() + 1) * 128
    return predict_frame.type(torch.uint8).numpy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
