import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import seaborn as sns

# Define STE function


def STE(frame):
    return sum([x**2 for x in frame]) / len(frame)


# Load audio file
unvoice = AudioSegment.from_file("318622__matucha__roomtone_aircon_01.wav")
voice = AudioSegment.from_file(
    "519189__inspectorj__request-42-hmm-i-dont-know.wav")


# Define frame parameters
frame_size = 2048  # Replace with your desired frame size in samples
hop_size = 2048  # Replace with your desired hop size in samples
# threshold = 0.3  # Replace with your desired threshold for STE classification

# Split audio into frames
uv_samples = np.array(unvoice.get_array_of_samples())
v_samples = np.array(voice.get_array_of_samples())
num_frames_uv = int(np.ceil(len(uv_samples) / hop_size))
num_frames_v = int(np.ceil(len(v_samples) / hop_size))
frames_uv = np.zeros((num_frames_uv, frame_size))
frames_v = np.zeros((num_frames_v, frame_size))

labeled_uv = np.array([0 for _ in range(num_frames_uv)])
labeled_v = np.array([1 for _ in range(num_frames_v)])
labeled = np.concatenate([labeled_uv, labeled_v])


# Lấy từng frame của samples
for i in range(num_frames_uv):
    frame_start = i * hop_size
    frame_end = frame_start + frame_size
    frame = uv_samples[frame_start:frame_end]
    frames_uv[i, : len(frame)] = frame

for i in range(num_frames_v):
    frame_start = i * hop_size
    frame_end = frame_start + frame_size
    frame = v_samples[frame_start:frame_end]
    frames_v[i, : len(frame)] = frame

# Calculate STE
ste_v = np.zeros(num_frames_v)
ste_uv = np.zeros(num_frames_uv)

for i in range(num_frames_uv):
    ste_uv[i] = STE(frames_uv[i, :])
# ste_uv = ste_uv / np.max(ste_uv)


for i in range(num_frames_v):
    ste_v[i] = STE(frames_v[i, :])
# ste_v = ste_v / np.max(ste_v)

ste = np.concatenate([ste_uv, ste_v])

ste = ste / np.max(ste)

# Hàm cost_function


# def cost_function(true_labels, predicted_labels):
#     # Tìm số khung bị phân loại sai
#     num_misclassified_frames = np.sum(true_labels != predicted_labels)

#     # Trả về số khung bị phân loại sai
#     return num_misclassified_frames


# threshold = [0.1 + i * 0.1 for i in range(10)]
# for i in threshold:
#     is_voiced = np.zeros(num_frames_uv + num_frames_v)
#     for j in range(num_frames_uv + num_frames_v):
#         is_voiced[j] = 1 if ste[j] > i else 0
#     # plt.figure()
#     # plt.plot(is_voiced)
#     print(i)
#     print(cost_function(labeled, is_voiced))


miss_frames = [] 

def cost_function(threshold):
    is_voiced = np.zeros(num_frames_uv + num_frames_v)
    for j in range(num_frames_uv + num_frames_v):
        is_voiced[j] = 1 if ste[j] > threshold else 0
    num_misclassified_frames = np.sum(labeled != is_voiced)
    miss_frames.append(num_misclassified_frames)
    return num_misclassified_frames


# Define the range of threshold values to search over
threshold_values = np.linspace(0, 1, num=1000)

# Find the threshold value that minimizes the cost function
min_threshold = np.argmin([cost_function(T) for T in threshold_values])

# Print the optimal threshold value and the corresponding cost
print("Optimal threshold value:", threshold_values[min_threshold])
print("Minimum cost:", cost_function(threshold_values[min_threshold]))


# Thử nghiệm với ngưỡng 0.1 xem phân loại đúng không

# Create time axis
# sample_rate = unvoice.frame
# time_axis = np.arange(len(is_voiced)) * hop_size / \
#     (num_frames_uv + num_frames_v)

# # Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# # Plot voiced/unvoiced frames
# ax1.stem(time_axis, is_voiced, use_line_collection=True)
# ax1.set_xlabel("Time (s)")
# ax1.set_ylabel("Voiced/Unvoiced")
# ax1.set_title("Classification of Audio Frames")

# # Plot STE
# ax2.plot(time_axis, energy)
# # ax2.xlabel("Time (s)")
# # ax2.ylabel("Short-Time Energy")
# # ax2.title("Short-Time Energy of Audio File")
# ax2.set_xlabel("Time (s)")
# ax2.set_ylabel("Short-Time Energy")
# ax2.set_title("Short-Time Energy of Audio File")

# plt.plot(time_axis, ste)
# plt.xlabel("Time (s)")
# plt.ylabel("Short-Time Energy")
# plt.title("Short-Time Energy of Audio File")

# plt.show()


audio = AudioSegment.from_file("studio_F2.wav")

# Split audio into frames
samples = np.array(audio.get_array_of_samples())
num_frames = int(np.ceil(len(samples) / hop_size))
frames = np.zeros((num_frames, frame_size))
for i in range(num_frames):
    frame_start = i * hop_size
    frame_end = frame_start + frame_size
    frame = samples[frame_start:frame_end]
    frames[i, :len(frame)] = frame

energy = np.zeros(num_frames)
for i in range(num_frames):
    energy[i] = STE(frames[i, :])
energy = energy / np.max(energy)

is_voiced = np.zeros(num_frames)
for j in range(num_frames):
    is_voiced[j] = 1 if energy[j] > threshold_values[min_threshold] else 0

sample_rate = audio.frame_rate
time_axis = np.arange(len(energy)) * hop_size / sample_rate

#plot figure
plt.plot(energy)
plt.xlabel("Time (s)")
plt.ylabel("Short-Time Energy")
plt.title("Short-Time Energy of Audio File")

plt.figure()
plt.stem(time_axis, is_voiced, use_line_collection=True)
plt.xlabel("Time (s)")
plt.ylabel("Voiced/Unvoiced")
plt.title("Classification of Audio Frames")

plt.figure()
plt.plot(samples)

plt.figure()
freq_miss, threshold, patches = plt.hist(miss_frames, 100, color='blue', edgecolor='black')

plt.figure()
threshold_values = np.insert(threshold_values, 0, 0)
plt.bar(miss_frames, threshold_values, width=10)



plt.show()
