import os
import cv2
from tqdm import tqdm

def is_video_file(filepath):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.m4v', '.3gp')
    return filepath.lower().endswith(video_extensions)

def save_frames_from_video(video_path, output_dir, step=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        if frame_count % step == 0:
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Total frames saved: {frame_count}")

def save_video(image_path):
    images = [img for img in os.listdir(image_path) if img.endswith(".png") or img.endswith(".jpg")]
    # for image in images:
    #     img_name = image.split('.')[0]
    #     img_name_id = img_name[6:].zfill(5)
    #     os.rename(os.path.join(image_path, image), os.path.join(image_path, f'frame_{img_name_id}.jpg'))
    images = sorted(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', etc. depending on your needs
    output_video = os.path.join(image_path, '../video.mp4',)
    fps = 30
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width = frame.shape[:2]
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for image in tqdm(images, total=len(images)):
        frame = cv2.imread(os.path.join(image_path, image))
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved as {output_video}")