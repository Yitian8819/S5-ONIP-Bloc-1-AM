import numpy as np
import sounddevice as sd
import base64
from matplotlib import pyplot as plt

# Extract data 提取数据
with open('/Users/hanyitian/Desktop/ONIP_Bloc1_AM/data_02.txt', 'rb') as file_to_decode:
    binary_file_data = file_to_decode.read()   # Read the file 读取文件
    signal_data = np.frombuffer(base64.b64decode(binary_file_data), np.int16)   # Extract the data 提取数据

# Some important information : It is an audio file of 24kHz / 16 bits
# 一些重要信息：这是一个音频文件 24kHz / 16bits

# Set the time axis 设置时间轴
sample_rate = 24000  # Sample rate - 24kHz 采样率为 24 kHz
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
fft_freq = np.fft.fftfreq(len(signal_data), d=samp)

# Draw the signal FFT waveform 绘制信号傅里叶变换波形
plt.figure(1)
plt.subplot(2,1,2)
plt.plot(fft_freq, fft_amp)
plt.title('FFT Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (V)')
plt.tight_layout()
plt.grid(True)

# Demodulation 解调
carrier_peaks = np.argsort(fft_amp)[-4:]    # Find the position of the peak in frequency field 找到频域最大峰值位置
carrier_freq = fft_freq[carrier_peaks]
center_freq = (carrier_freq[0]+carrier_freq[3])/2   # Determine the frequency of the original carrier signal 确定原始载波频率
print("The center frequency is :",center_freq,"Hz")
dem_freq = 6000   # Set the frequency at 6000 Hz 将频率设定为6000Hz
carrier_signal = np.cos(2 * np.pi * dem_freq * time)   # Establish demodulated signal 创建解调信号
dem_signal = signal_data * carrier_signal   # Demodulation 解调信号

# FFT demodulation 解调信号的傅里叶变换
fft_dem_signal = np.fft.fft(dem_signal)
fft_dem_amp = np.abs(fft_dem_signal)
fft_dem_freq = np.fft.fftfreq(len(dem_signal), d=samp)

# Draw the FFT demodulation 绘制解调信号FFT图形
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(fft_dem_freq, fft_dem_amp)
plt.title("Demodulated FFT Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")

# Design a low-pass filter 设计低通滤波器
def fft_lowpass_filter(data, cutoff, fs):
    fft_data = np.fft.fft(data)   # FFT 傅里叶变换
    freq = np.fft.fftfreq(len(data), d = 1 / fs)
    # Set the components exceeding the cutoff frequency to zero 将超过截止频率的分量置零
    fft_data[np.abs(freq) > cutoff] = 0
    filtered_data = np.fft.ifft(fft_data)   # FFT inverse 傅立叶逆变换
    return np.real(filtered_data)

fs = sample_rate   # Sample frequency 采样频率
cutoff = center_freq   # Cutoff frequency 截止频率
baseband_signal = fft_lowpass_filter(dem_signal, cutoff, fs)   # Filtered signal 滤波信号

# Play the audio 播放音频
volume = 0.001
play_signal = dem_signal * volume   # Prevent the audio too loud 防止音频响度过大
sd.play(play_signal, sample_rate)
sd.wait()
print("The text in the audio is : \" Hello! You manage to read this message. But you still have a final message to decode. Good luck ! \" ")

# Draw the demodulated signal 绘制解调信号
plt.figure(2)
plt.subplot(2,1,2)
plt.plot(time, baseband_signal)
plt.title("Demodulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.tight_layout()
plt.show()