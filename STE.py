import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

import os
print(os.getcwd())

# Define STE function


def STE(frame):
    return sum([x**2 for x in frame]) / len(frame)


# Load audio file
audio = AudioSegment.from_file("Atest.mp3")

# Define frame parameters
frame_size = 2048  # Replace with your desired frame size in samples
hop_size = 512  # Replace with your desired hop size in samples
threshold = 0.3  # Replace with your desired threshold for STE classification

# Split audio into frames
samples = np.array(audio.get_array_of_samples())
num_frames = int(np.ceil(len(samples) / hop_size))
frames = np.zeros((num_frames, frame_size))
for i in range(num_frames):
    frame_start = i * hop_size
    frame_end = frame_start + frame_size
    frame = samples[frame_start:frame_end]
    frames[i, : len(frame)] = frame

# Calculate STE and classify frames as voiced or unvoiced
ste = np.zeros(num_frames)
for i in range(num_frames):
    ste[i] = STE(frames[i, :])
ste = ste / np.max(ste)

is_voiced = np.zeros(num_frames)
for i in range(num_frames):
    is_voiced[i] = ste[i] > threshold

# Calculate STE of each frame
energy = np.zeros(num_frames)
for i in range(num_frames):
    energy[i] = STE(frames[i, :])


# Create time axis
sample_rate = audio.frame_rate
time_axis = np.arange(len(is_voiced)) * hop_size / sample_rate

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# Plot voiced/unvoiced frames
ax1.stem(time_axis, is_voiced, use_line_collection=True)
# ax1.xlabel("Time (s)")
# ax1.ylabel("Voiced/Unvoiced")
# ax1.title("Classification of Audio Frames")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voiced/Unvoiced")
ax1.set_title("Classification of Audio Frames")

# Plot STE
ax2.plot(time_axis, energy)
# ax2.xlabel("Time (s)")
# ax2.ylabel("Short-Time Energy")
# ax2.title("Short-Time Energy of Audio File")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Short-Time Energy")
ax2.set_title("Short-Time Energy of Audio File")

plt.show()
