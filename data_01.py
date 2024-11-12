import numpy as nu
import pandas as pd
from matplotlib import pyplot as plt

# Extract data 提取数据
df = pd.read_table('/Users/hanyitian/Desktop/ONIP_Bloc1_AM/data_01.csv')   # Read the file 读取文件
df_sp = df['#CHANNEL:CH1'].str.split(',', expand=True)   #use "," to divide in 3 columns 利用逗号将数据分隔为三列
df_sp = df_sp.dropna()   # Delete unsuccessful lines 删除失败行
df_sp.columns = ['Index', 'Time', 'Volt']   # Name the columns 命名列
df_sp = df_sp[df_sp['Time'] != 'Time(s)']   # Delete the header 删除表头
df_sp['Time'] = df_sp['Time'].astype(float)
df_sp['Volt'] = df_sp['Volt'].astype(float)   # Convert data into floating point 转换为浮点数

# Draw the original signal waveform 绘制原始信号波形
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(df_sp['Time'], df_sp['Volt'])
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)

# Signal FFT 信号傅里叶变换
time = df_sp['Time']
signal = df_sp['Volt']
samp = time.iloc[1] - time.iloc[0]   # Determine the sampling interval 确定采样区间
fft_signal = nu.fft.fft(signal)
fft_amp = nu.abs(fft_signal)
fft_freq = nu.fft.fftfreq(len(signal), d=samp)

# Draw the signal FFT waveform 绘制信号傅里叶变换波形
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(fft_freq, fft_amp)
plt.title('FFT Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (V)')
plt.grid(True)

# Demodulation 解调
carrier_peak = nu.argmax(fft_amp)   # Find the position of the peak in frequency field 找到频域最大峰值位置
carrier_freq = nu.abs(fft_freq[carrier_peak])   # Determine the frequency of the original carrier signal 确定原始载波频率
carrier_signal = nu.cos(2 * nu.pi * carrier_freq * time)   # Establish demodulated signal 创建解调信号
dem_signal = signal * carrier_signal   # Demodulation 解调信号

# FFT demodulation 解调信号的傅里叶变换
fft_dem_signal = nu.fft.fft(dem_signal)
fft_dem_amp = nu.abs(fft_dem_signal)
fft_dem_freq = nu.fft.fftfreq(len(dem_signal), d=samp)

# Draw the FFT demodulated signal 绘制解调信号的傅里叶频谱图
plt.figure(2)
plt.subplot(2,1,2)
plt.plot(fft_dem_freq, fft_dem_amp)
plt.title("Demodulated FFT Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.tight_layout()

# Design a low-pass filter 设计低通滤波器
def fft_lowpass_filter(data, cutoff, fs):
    fft_data = nu.fft.fft(data)   # FFT 傅里叶变换
    freq = nu.fft.fftfreq(len(data), d=1 / fs)
    # Set the components exceeding the cutoff frequency to zero 将超过截止频率的分量置零
    fft_data[nu.abs(freq) > cutoff] = 0
    filtered_data = nu.fft.ifft(fft_data)   # FFT inverse 傅立叶逆变换
    return nu.real(filtered_data)

fs = 1 / samp   # Sample frequency 采样频率
cutoff = carrier_freq   # Cutoff frequency 截止频率
baseband_signal = fft_lowpass_filter(dem_signal, cutoff, fs)   # Filtered signal 滤波信号

# Draw the demodulated signal 绘制解调信号
plt.figure(1)
plt.subplot(2,1,2)
plt.plot(time, baseband_signal)
plt.title("Demodulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.tight_layout()
plt.show()

# Save as a csv file 保存为csv文件
df_sp.to_csv('/Users/hanyitian/Desktop/ONIP_Bloc1_AM/data_01_modulated.csv', index=False)