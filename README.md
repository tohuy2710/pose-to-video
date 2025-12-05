# pose-to-video

This repo is simplify from https://github.com/sign-language-processing/pose-to-video for the purpose that brings into production code.

A small utility to generate a video from a pose sequence using a pix2pix generator model.

This project reads a `.pose` file (custom pose format), renders pose frames, feeds them into
a trained pix2pix generator, and saves the output as an MP4 video.

## Features

- Convert pose sequences to video frames using a pix2pix model
- Simple Python API and a CLI-like entrypoint (`main.py`)
- Minimal dependencies (OpenCV, TensorFlow, NumPy)

## File structure

- `main.py` — example runner that calls the high-level function
- `pose_to_video.py` — core logic: load pose, run model, write video
- `requirements.txt` — Python dependencies
- `pix_to_pix.h5` (not included) — trained generator model (see download instructions)

## Requirements

- Python 3.8+ (tested with Python 3.8/3.9/3.10)
- A working GPU or CPU TensorFlow build compatible with the model
- See `requirements.txt` for exact package versions. Install with:

```bash
pip install -r requirements.txt
```

## Download the model

The repository does not include the trained model. Download the generator model and save it as `pix_to_pix.h5` in the project root. Example (use this exact command to save the file):

```bash
wget "https://firebasestorage.googleapis.com/v0/b/sign-mt-assets/o/models%2Fgenerator%2Fmodel.h5?alt=media" -O pix_to_pix.h5
```

Make sure the downloaded file is named `pix_to_pix.h5` and is present in the project folder before running the scripts.

## Usage

1. Place your `.pose` file (for example `1.pose`) in the project folder.
2. Download and place `pix_to_pix.h5` in the project root (see previous section).
3. Run the example runner:

```bash
python3 main.py
```

By default `main.py` runs:

```python
from pose_to_video import generate_video_from_pose

generate_video_from_pose("1.pose", "output_pix2pix.mp4")
```

Or call the function directly from your own script:

```python
from pose_to_video import generate_video_from_pose

generate_video_from_pose("path/to/input.pose", "out.mp4", model_file="pix_to_pix.h5")
```

## Model notes

- The code expects a Keras/TensorFlow `.h5` model that can be loaded with `tensorflow.keras.models.load_model`.
- The generator should accept 256x256 pose images (RGB) and output 256x256 RGB frames.
- If you get errors loading the model, ensure your TensorFlow version matches the version used to save the model.

## Troubleshooting

- FileNotFoundError: check the paths to your `.pose` file and `pix_to_pix.h5`.
- Incompatible model: try a different TF version or re-save the model with your current TensorFlow.
- Video looks black or distorted: confirm the generator outputs 0-255 uint8 RGB images of shape (256,256,3).
- If OpenCV cannot write the file, confirm you have `ffmpeg` or a compatible codec installed on your system.

## License

MIT License — see `LICENSE` (or consider this project under the MIT terms).

---


