import numpy as np
import sounddevice as sd
import base64
from matplotlib import pyplot as plt

from data_01 import dem_signal

# Extract data 提取数据
with open('/Users/hanyitian/Desktop/ONIP_Bloc1_AM/data_03.txt', 'rb') as file_to_decode:
    binary_file_data = file_to_decode.read()   # Read the file 读取文件
    signal_data = np.frombuffer(base64.b64decode(binary_file_data), np.int16)   # Extract the data 提取数据

# Some important information : It is an audio file of 160kHz / 16 bits
# 一些重要信息：这是一个音频文件 160kHz / 16bits

# Set the time axis 设置时间轴
sample_rate = 160000  # Sample rate - 160kHz 采样率为 160 kHz
time = np.arange(len(signal_data)) / sample_rate

# Draw the original signal waveform 绘制原始信号波形
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time, signal_data)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)

samp = 1 / sample_rate
fft_signal = np.fft.fft(signal_data)
fft_amp = np.abs(fft_signal)
fft_freq = np.fft.fftfreq(len(signal_data), d = samp)

# Draw the signal FFT waveform 绘制信号傅里叶变换波形
plt.figure(1)
plt.subplot(2,1,2)
plt.plot(fft_freq, fft_amp)
plt.title('FFT Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (V)')
plt.tight_layout()
plt.grid(True)

# Find center frequencies 寻找中心频率
center_freqs = []
fft_amp_copy = fft_amp.copy()   # Copy the signal information 复制信号信息
fft_amp_copy[fft_freq < 0] = 0   # Take the positive frequency component 取频率正值部分
num_peaks = 3
for i in range(num_peaks):
    peak = np.argmax(fft_amp_copy)   # Take the maximum peak value 取最大峰值
    center_freq = fft_freq[peak]
    center_freqs.append(center_freq)   # Take the center frequency 取中心频率
    start = max(0, peak - 5000)
    end = min(len(fft_amp_copy), peak + 5000)
    fft_amp_copy[start:end] = 0   # Determine the zeroing interval 确定清零区间
print("The center frequencies are :", center_freqs)
center_freqs = [12000,24000,36000]   # Manually input the approximate frequency value 手动输入近似频率值

# Design a low-pass filter 设计低通滤波器
def fft_lowpass_filter(data, cutoff, fs):
    fft_data = np.fft.fft(data)   # FFT 傅里叶变换
    freq = np.fft.fftfreq(len(data), d = 1 / fs)
    # Set the components exceeding the cutoff frequency to zero 将超过截止频率的分量置零
    fft_data[np.abs(freq) > cutoff] = 0
    filtered_data = np.fft.ifft(fft_data)   # FFT inverse 傅立叶逆变换
    return np.real(filtered_data)

# Demodulation 解调
dem_signals = []
cutoff = 6000   # Set the cutoff frequency 设定截止频率
for freq in center_freqs:
    carrier = np.cos(2 * np.pi * freq * np.arange(len(signal_data)) / sample_rate)
    dem = signal_data * carrier   # Multiply separately with the three different frequency carriers 分别与三种不同频率载波相乘
    dem_filtered = fft_lowpass_filter(dem, cutoff, sample_rate)   # Low-pass filtering 低通滤波
    dem_signals.append(dem_filtered)

# Play the audio 播放音频
volume = 0.01
play_signal1 = dem_signals[0] * volume   # Prevent the audio too loud 防止音频响度过大
sd.play(play_signal1, sample_rate)
sd.wait()
play_signal2 = dem_signals[1] * volume   # Prevent the audio too loud 防止音频响度过大
sd.play(play_signal2, sample_rate)
sd.wait()
play_signal3 = dem_signals[2] * volume   # Prevent the audio too loud 防止音频响度过大
sd.play(play_signal3, sample_rate)
sd.wait()

# Draw the demodulated signal 绘制解调信号
plt.figure(2)
plt.subplot(3,1,1)
plt.plot(time, dem_signals[0])
plt.title("Reconstruct Signal 1")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(time, dem_signals[1])
plt.title("Reconstruct Signal 2")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(time, dem_signals[2])
plt.title("Reconstruct Signal 3")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.tight_layout()
plt.show()