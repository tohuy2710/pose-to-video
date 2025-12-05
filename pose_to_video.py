import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

# ------------------------------------------------------------
#  PIX2PIX: Translate Image
# ------------------------------------------------------------
def translate_image(model, image_rgb):
    pixels = image_rgb.astype(np.float32) / 255.0
    pixels = (pixels - 0.5) * 2.0  # normalize [-1, 1]

    tensor = np.expand_dims(np.expand_dims(pixels, 0), 0)  # (1,1,256,256,3)
    tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)

    pred = model(tensor, training=True).numpy()
    pred = (pred * 0.5) + 0.5
    pred = pred * 255.0

    pred = np.squeeze(pred, 0)
    pred = np.squeeze(pred, 0)

    return pred.astype(np.uint8)


# ------------------------------------------------------------
#  POSE → IMAGE GENERATOR
# ------------------------------------------------------------
def pose_to_video_frames(pose: Pose, model):
    scale_w = pose.header.dimensions.width / 256
    scale_h = pose.header.dimensions.height / 256

    pose.body.data /= np.array([scale_w, scale_h, 1])
    pose.header.dimensions.width = pose.header.dimensions.height = 256

    visualizer = PoseVisualizer(pose, thickness=1)

    for pose_img_bgr in visualizer.draw():
        pose_img_rgb = cv2.cvtColor(pose_img_bgr, cv2.COLOR_BGR2RGB)
        yield translate_image(model, pose_img_rgb)


# ------------------------------------------------------------
#  MAIN FUNCTION
# ------------------------------------------------------------
def generate_video_from_pose(pose_file, output_mp4, model_file="pix_to_pix.h5"):
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"No such directory: {pose_file}")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No such directory: {model_file}")

    # Load pose
    with open(pose_file, "rb") as f:
        pose = Pose.read(f.read())

    fps = pose.body.fps if pose.body.fps > 0 else 30
    w = h = 256

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))

    print("Loading pix2pix model...")
    model = load_model(model_file, compile=False)

    print("Model loaded.")
    print("Generating video from pose...")
    for frame in pose_to_video_frames(pose, model):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_out.write(frame_bgr)

    video_out.release()
    print("Done! Video saved at:", output_mp4)
