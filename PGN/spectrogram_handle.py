import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# 1. 加载音频文件
# 将 'bird_sound.wav' 替换为你的鸟鸣声音频文件路径
sample_rate, audio_data = wavfile.read('/dev/sda1/dataset/BirdsData_16000/0009/111651_2.wav')

# 如果是双声道音频，提取单声道数据
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)  # 转换为单声道（取平均）

# 2. 设置频谱图参数
window_size = 1024          # 窗口大小（控制频率分辨率）
overlap = window_size // 2  # 窗口重叠长度
nfft = window_size          # FFT 点数

# 3. 计算频谱图
frequencies, times, spectrogram_data = spectrogram(
    audio_data, fs=sample_rate, window='hamming', nperseg=window_size,
    noverlap=overlap, nfft=nfft
)

# 4. 绘制频谱图
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='auto', cmap='jet')
plt.ylabel('频率 (Hz)')
plt.xlabel('时间 (秒)')
plt.title('鸟鸣声频谱图')
plt.colorbar(label='强度 (dB)')
plt.ylim(0, 10000)  # 可选：限制频率范围（例如 0-10kHz）
plt.show()