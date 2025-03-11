###################
# 我们一步一步来。现在  是空的。我想定义AP类，其生成信号x(t) = Ae^jωt 
###################




import numpy as np
import matplotlib.pyplot as plt
import os, shutil
import csv

class AccessPoint:
    def __init__(self, ap_id, num_antennas=4, freq=900e6, output_dir="sim_figures/AP"):
        self.ap_id = ap_id
        self.num_antennas = num_antennas
        self.freq = freq  # 900 MHz
        self.wavelength = 3e8 / freq  # 波长计算
        self.antenna_spacing = self.wavelength / 2  # 半波长间距
        self.sweep_duration = 0.05  # 50ms per phase (论文参数)
        self.sample_rate = 1000  # 1kHz采样率
        self.output_dir = os.path.join(output_dir, f"AP{ap_id}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 根据论文定义前导码
        self.preamble = {
            1: [1,0,1,0,1,0,1,0],  # AP1前导码
            2: [1,1,0,0,1,1,0,0]   # AP2前导码
        }[ap_id]

    def generate_signal(self, duration=1.0):
        """生成包含前导码和相位扫描的完整信号"""
        # 时间轴
        total_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, total_samples)
        
        # 生成前导码信号
        preamble_samples = len(self.preamble) * int(0.1 * self.sample_rate)
        preamble_signal = np.concatenate(
            [np.ones(int(0.1*self.sample_rate)) if bit else np.zeros(int(0.1*self.sample_rate)) 
             for bit in self.preamble])
        
        # 计算剩余时间用于相位扫描
        remaining_duration = duration - (len(self.preamble)*0.1)
        phase_sweep = np.linspace(-np.pi, np.pi, int(remaining_duration/self.sweep_duration))
        
        # 生成扫描信号
        sweep_samples = total_samples - preamble_samples
        sweep_signal = np.zeros(sweep_samples)
        theta_values = []
        
        # 添加多径效应（论文第3.2节）
        multipath_ratio = 0.3  # NLOS路径信号强度比例
        delay = 0.01  # 多径时延
        
        for theta in phase_sweep:
            # 多天线相位设置（论文Algorithm 1）
            phases = [(i-1)*np.pi*np.sin(theta) for i in range(1, self.num_antennas+1)]
            
            # 信号合成（简化版，实际应包含多径效应）
            combined_signal = np.sum([np.exp(1j*(2*np.pi*self.freq*t + phase)) 
                                    for phase in phases], axis=0)
            
            # 添加多径信号（论文公式扩展）
            multipath_signal = np.sum([
                np.exp(1j*(2*np.pi*self.freq*(t-delay) + phase + np.random.uniform(0, 2*np.pi))) 
                for phase in phases
            ], axis=0) * multipath_ratio
            
            combined_signal += multipath_signal
            amplitude = np.abs(combined_signal)  # 包络检测
            
            # 添加噪声模拟实际环境
            amplitude += np.random.normal(0, 0.1, amplitude.shape)
            
            sweep_signal += amplitude.tolist()
            theta_values.extend([theta]*len(amplitude))
        
        # 确保信号长度正确
        full_signal = np.concatenate([preamble_signal, sweep_signal])[:total_samples]
        theta_values = theta_values[:total_samples]
        
        return t, full_signal, theta_values

    def save_and_plot(self, t, signal, theta):
        """保存数据和绘图"""
        # 保存CSV
        with open(os.path.join(self.output_dir, 'signal_data.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Time(s)', 'Amplitude(V)', 'Phase(rad)'])
            for time, amp, phase in zip(t, signal, theta):
                writer.writerow([round(time,4), round(amp,4), round(phase,4)])
        
        # 绘制信号波形
        plt.figure(figsize=(12, 6))
        plt.plot(t, signal, label='Received Amplitude')
        plt.title(f"AP{self.ap_id} Signal Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'signal_waveform.png'))
        plt.close()
        
        # 绘制相位-振幅关系
        plt.figure(figsize=(8,6))
        plt.scatter(theta, signal, s=2, alpha=0.5)
        plt.title(f"AP{self.ap_id} Phase-Amplitude Relationship")
        plt.xlabel("Phase Shift θ (rad)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'phase_amplitude.png'))
        plt.close()

if __name__ == "__main__":
    try:
        # 清理旧数据
        if os.path.exists("sim_figures/AP"):
            shutil.rmtree("sim_figures/AP")
        
        # 生成信号
        for ap_id in [1, 2]:
            ap = AccessPoint(ap_id)
            t, signal, theta = ap.generate_signal(duration=1.0)
            ap.save_and_plot(t, signal, theta)
        
        print("AP信号生成成功！请检查 sim_figures/AP 文件夹")
    except Exception as e:
        print(f"生成失败：{str(e)}")
        print("可能原因：")
        print("1. 请确保有文件写入权限")
        print("2. 检查matplotlib是否安装正确")
        print("3. 尝试手动创建 sim_figures 文件夹") 
