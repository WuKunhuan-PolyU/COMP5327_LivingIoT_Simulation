"""
Paper: Living IoT:AFlyingWireless Platformon LiveInsects
Bee flight simulation with a focus on the localization with multiple APs
"""
import numpy as np
import os, datetime, math, shutil, random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

if os.path.exists('sim.gif'):
    os.remove('sim.gif')
if os.path.exists('sim_figures'):
    shutil.rmtree('sim_figures')


# Environment and Display Parameters
NUM_BEES = 10
NUM_FOOD_SOURCES = 5  # randomly distributed in the field
HIVE_POS = (0.5, 0.1)                  # normalized
AP_POS = [(0.25, 0.5), (0.75, 0.5)]    # normalized
FIG_SIZE = (8, 8)
NECTAR_DISPLAY_SIZE = 50
NECTAR_EXPECTED_STORAGE = 15
NECTAR_EXPECTED_RECOVERY_TIME = 20

# AP Parameters
AP_PREAMBLES = [(1,0,1,0,1,0,1,0), (1,1,0,0,1,1,0,0)]
# numbers are chosen for the ease of simulation
AP_SWEEP_T = 50           # time (ms) per sweep, including preamble <- from the paper
AP_SWEEP_PREAMBLE_T = 5   # time (ms) of preamble in a sweep <- self chosen
AP_PHASESHIFT_NUM = 90    # number of phase shifts beside preamble per sweep <- self chosen
# in (50, 5, 90) setting, each phase shift takes 0.5ms
# as TDMA is used, the signals are transmitted in a time-division manner
# for example, if there are 2 APs, and the sweep time is 50ms
# the first 25ms, only AP0 is transmitting, 
# the second 25ms, only AP1 is transmitting
AP_NUM_ANTENNAS = 4
AP_FREQ = 915e6         # 915 MHz
AP_SIGNAL_STRENGTH = 28 # dBm

# Simulation Parameters
FRAME_TIME = 50    # ms
SIM_TIME = 2000   # ms
NUM_FRAMES = round(SIM_TIME / FRAME_TIME)

# Bee Parameters
# bumblebee movement speed: x m/s (normally)
# which is (x / FIELD_SIZE) * normalized_edge_length m/s
# 1s = 20 frames
# 1 frame = (x / FIELD_SIZE / 20) * normalized_edge_length m
FIELD_SIZE = 100         # m
FIELD_ASPECT_RATIO = 1.0 # width / height
# Distance between two APs is 50, close to the paper configuration
BUMBLEBEE_SPEED = 5    # m/s
STEP_SIZE = BUMBLEBEE_SPEED / FIELD_SIZE / 20
MIN_STEP = 0.75 * STEP_SIZE 
MAX_STEP = 1.25 * STEP_SIZE
STEP_NOISE = 0.25 * STEP_SIZE
NECTAR_DISCOVERY_DISTANCE = 5 * STEP_SIZE
NECTAR_EMPTY_DISCOVERY_DISTANCE = 10 * STEP_SIZE
OVERLAP_DISTANCE_ERROR = 3 * STEP_SIZE
MOVEMENT_DISAPPEAR_TIME = 10000 # ms

# 新增全局参数
AP_SAMPLE_RATE = 40 * AP_FREQ  # 40倍过采样
AP_SIGNAL_AMPLITUDE = 10**(AP_SIGNAL_STRENGTH / 20)  # 转换dBm为线性振幅

# Class definitions
class Food:
    def __init__(self, position):
        self.position = position
        self.nectar = round((0.8 + 0.4 * random.random()) * NECTAR_EXPECTED_STORAGE)
        self.recovery_timer = 0

class Bee:
    def __init__(self, food_sources):
        self.position = np.array(HIVE_POS)
        self.target_food = None
        self.food_sources = food_sources
        self.has_food = False
        self.known_empty_sources = set() # Local memory of empty food sources
        self.history = []  # Record position and time
        self.signal_memory = []  # 存储最近3个周期的信号数据

    def reset_knowledge(self):
        self.known_empty_sources.clear()

    def update_position_history(self, frame_time):
        self.history.append((self.position.copy(), frame_time))
        if len(self.history) > math.ceil(MOVEMENT_DISAPPEAR_TIME / FRAME_TIME):
            self.history.pop(0)

    def receive_signal(self, ap_id, timestamps, samples):
        """存储接收到的信号数据"""
        if len(self.signal_memory) >= 3:
            self.signal_memory.pop(0)
        self.signal_memory.append({
            'ap_id': ap_id,
            'timestamps': timestamps,
            'samples': samples
        })
    
    def sample_position_history(self):
        pass
    
    def get_estimated_position(self):
        pass

class AP:
    def __init__(self, ap_id, 
                 location, # (x (in m), y (in m))
                 antenna_num, # assume the antennas are distributed in a line, and the center is the location
                 frequency, # in Hz
                 preamble_signal, 
                 antenna_direction = 0, # radians, counterclockwise from the positive x-axis
                 sweep_time = 50, # in ms
                 preamble_duration = 5, # in ms
                 phase_shifts_per_sweep = 90,  # 根据论文Algorithm 1命名
                 ):
        self.ap_id = ap_id
        self.location = np.array(location)
        self.preamble = preamble_signal

        # used to calculate the signal
        self.signal_start = 0  # in ms
        self.freq = frequency  # in Hz

        self.antenna_num = antenna_num
        self.antenna_direction = antenna_direction
        self.wavelength = 3e8 / frequency  # Speed of light / frequency, in meters
        
        self.antenna_spacing = self.wavelength / 2  # Half wavelength spacing, in meters, should be around 33cm for 915MHz
        antenna_direction_vector = np.array([np.cos(antenna_direction), np.sin(antenna_direction)])
        self.antenna_locations = [
            self.location + self.antenna_spacing * (i - (self.antenna_num-1)/2) * antenna_direction_vector
            for i in range(self.antenna_num)
        ]
        
        self.sweep_time = sweep_time
        self.preamble_duration = preamble_duration

        # 波束成形相关参数
        self.current_theta = -np.pi/2  # 当前波束角度
        self.phase_shifts_per_sweep = phase_shifts_per_sweep 
        self.phase_step = np.pi / self.phase_shifts_per_sweep  # δ = π/90 ≈ 2°
        self.antenna_phases = [0.0] * (self.antenna_num + 1)  # 天线编号从1开始

        # Run the tests
        self.test_signal()
        self.signal_at_position((0.0, 0.0))

    def calculate_attenuation(self, position):
        """根据位置获取总衰减"""
        x = position[0] * FIELD_SIZE
        y = position[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
        with open(f'sim_stats/AP_{self.ap_id}/nlos_attenuation.csv') as f:
            for line in f:
                if line.startswith(f"{int(x)},{int(y)}"):
                    return float(line.split(',')[2])
        return 0  # 默认无衰减
    
    def transmit_signal(self, bees, current_time):
        """改进的TDMA信号传输"""
        # 重置波束角度
        self.current_theta = -np.pi/2  # 每次传输都从-π/2开始
        
        # 每个相位步长持续时间
        phase_duration = (self.sweep_time - self.preamble_duration) / self.phase_shifts_per_sweep
        
        # 分阶段生成信号
        timestamps = []
        samples = []
        
        # 前导码阶段
        preamble_ts = np.arange(current_time, current_time+self.preamble_duration, 1000/AP_SAMPLE_RATE)
        timestamps.extend(preamble_ts)
        samples.extend([self.sample_signal(t) for t in preamble_ts])
        
        # 相位扫描阶段
        for i in range(self.phase_shifts_per_sweep):
            # 更新波束角度
            self.current_theta += self.phase_step
            self.current_theta = np.clip(self.current_theta, -np.pi/2, np.pi/2)
            
            # 生成当前角度下的信号
            phase_ts = np.arange(
                current_time + self.preamble_duration + i*phase_duration,
                current_time + self.preamble_duration + (i+1)*phase_duration,
                1000/AP_SAMPLE_RATE
            )
            timestamps.extend(phase_ts)
            samples.extend([self.sample_signal(t) for t in phase_ts])
        
        # 发送给所有蜜蜂
        for bee in bees:
            attenuation = self.calculate_attenuation(bee.position)
            amp_scale = AP_SIGNAL_AMPLITUDE * 10**(-attenuation/20)
            bee.receive_signal(self.ap_id, timestamps, [s*amp_scale for s in samples])

    def calculate_beamforming_phase(self, position):
        """根据论文公式计算波束成形相位差"""
        # 计算到达角θ（相对于天线阵列方向）
        vec = position - self.location
        theta = np.arctan2(vec[1], vec[0]) - self.antenna_direction
        theta = np.clip(theta, -np.pi/2, np.pi/2)
        
        # 根据论文公式设置相位（天线编号从2开始）
        for j in range(2, self.antenna_num+1):
            self.antenna_phases[j-1] = (j-1) * np.pi * np.sin(theta)
        
        return sum(self.antenna_phases[1:])  # 合成相位（忽略0号天线）

    def sample_signal(self, timestamp, position=None):
        """生成带波束成形效果的信号
        信号组成：全局相位偏移 + 波束成形相位差
                  ┌─────────────┐
        时间 ──→ │ 相位调制器  │──→ 合成信号
                  └─────┬───┬───┘
                    全局偏移 │
                           波束成形相位差
        """
        # 基础相位偏移（TDMA调度引起）
        phase_shift = np.radians(self.current_theta)
        
        if position is not None:
            delta_phase = self.calculate_beamforming_phase(position)
            phase_shift += delta_phase
            # 记录波束成形相位差（用于调试）
            self.last_beamforming_phase = delta_phase
        
        # 原始信号生成
        if (self.signal_start is None or timestamp < self.signal_start):
            return 0
        elif (timestamp < self.signal_start + self.preamble_duration): 
            return self.preamble[int((timestamp - self.signal_start) / self.preamble_duration * len(self.preamble))]
        else: 
            time = (timestamp - self.signal_start - self.preamble_duration)
            return np.sin(2 * np.pi * self.freq * (time * 1e-3) + phase_shift)
    
    def sample_signal_range(self, 
                            start, # in ms
                            end,   # in ms
                            step,  # in ms
                            location = None # (x, y) in m
                            ): 
        '''
        Given a timestamp, return the sampled signal amplitude
        '''
        all_timestamps = np.arange(start, end, step)
        samples = [self.sample_signal(t, location) for t in all_timestamps]
        return samples
    
    def test_signal(self):
        """
        Generate dual-view signal analysis with separated time scales
        Save to sim_figures/AP_{ap_id}/test_signal.png
        """
        # Find the signal period
        period_ns = 1e9 / self.freq  # in ns
        preamble_duration_ms = self.preamble_duration  # in ms
        signal_duration_ns = 3 * period_ns  # in ns
        
        # Dynamic sampling settings
        samples_preamble = 100  # total preamble samples
        samples_per_period = 1000  # samples per period
        
        # Generate test signal data
        preamble_t = np.linspace(0, preamble_duration_ms, samples_preamble)
        signal_t = np.linspace(preamble_duration_ms, 
                             preamble_duration_ms + signal_duration_ns * 1e-6,  # in ns to ms
                             3 * samples_per_period)
        preamble_data = [self.sample_signal(t) for t in preamble_t]
        signal_data = [self.sample_signal(t) for t in signal_t]

        # Create dual-view layout
        _ = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, 
                             height_ratios=[1, 2],
                             hspace=0.3)
        
        # Preamble view (stepped line chart)
        ax1 = plt.subplot(gs[0])
        ax1.step(preamble_t, preamble_data, 
                where='post', 
                color='#FF4500',
                linewidth=1.5)
        ax1.set_title(f"Preamble Digital Signal ({preamble_duration_ms}ms)", pad=6)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Digital Level")
        ax1.grid(True, axis='y', linestyle=':')
        ax1.set_xlim(-0.1, preamble_duration_ms + 0.1)
        ax1.set_ylim(-0.1, 1.1)
        
        # Signal period view (ns scale)
        ax2 = plt.subplot(gs[1])
        relative_t = (signal_t - preamble_duration_ms) * 1e6  # in ns
        ax2.plot(relative_t, signal_data, 'b-', alpha=0.8)
        for i in range(3):
            t = (i+1)*period_ns
            ax2.axvline(t, color='green', linestyle='--', alpha=0.6)
            ax2.text(t-period_ns/2, 0.9, 
                    f'Cycle {i+1}\n({period_ns:.2f}ns)', 
                    ha='center', fontsize=9)
        
        # Create the overall chart
        ax2.set_title(f"Sinusoidal Wave Detail ({self.freq/1e6}MHz)", pad=6)
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, which='both', alpha=0.4)
        ax2.set_xlim(-period_ns*0.1, signal_duration_ns*1.1)
        ax2.set_ylim(-1.1, 1.1)
        output_dir = f'sim_figures/AP_{self.ap_id}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "test_signal.png"), 
                  dpi=300, bbox_inches='tight')
        plt.close()
        print (f"\nAccess Point {self.ap_id} Statistics:")
        print (f"├─ Location: {[round(i, 3) for i in self.location]}")
        print (f"├─ Preamble Duration: {preamble_duration_ms}ms")
        print (f"├─ Carrier Frequency: {self.freq/1e6}MHz")
        print (f"├─ Single Period: {period_ns:.4f}ns")
        print (f"├─ Wavelength: {self.wavelength:.4f}m")
        print (f"├─ Antenna Spacing: {self.antenna_spacing:.4f}m (half wavelength)")
        print (f"├─ Antenna Direction: {self.antenna_direction}rad")
        for i in range(self.antenna_num):
            print (f"│   {'├─' if i < self.antenna_num - 1 else '└─'} Antenna {i} Location: {[round(i, 3) for i in self.antenna_locations[i]]}")
        print (f"└─ 3 Cycles Duration: {signal_duration_ns:.2f}ns")
    
    def set_signal_start(self, timestamp):
        '''
        Set the signal start time
        => It will affect the signal generation
        '''
        self.signal_start = timestamp # in ms

    def get_signal_strength(self, position):
        """
        根据位置获取接收信号强度
        position: 归一化坐标 (0-1, 0-1)
        返回：dBm
        """
        # 转换为米坐标
        x_m = position[0] * FIELD_SIZE
        y_m = position[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
        
        # 获取预计算衰减值
        grid_x = int(round(x_m))
        grid_y = int(round(y_m))
        attenuation = self.attenuation_map.get((grid_x, grid_y), 0.0)
        
        # 计算接收功率
        return AP_SIGNAL_STRENGTH - attenuation

    def signal_at_position(self, norm_pos):
        """在指定位置验证信号特征"""
        position = np.array(norm_pos)
        
        # 生成时间序列（一个sweep周期）
        timestamps = np.arange(0, self.sweep_time, 0.1)  # 0.1ms分辨率
        
        # 计算各分量
        original = []
        attenuated = []
        phases = []
        
        for t in timestamps:
            # 原始信号（无衰减）
            amp = self.sample_signal(t, position=None)  # 禁用波束成形
            original.append(amp)
            
            # 衰减计算
            attenuation = self.calculate_attenuation(position)
            amp_scale = AP_SIGNAL_AMPLITUDE * 10**(-attenuation/20)
            
            # 实际接收信号（含波束成形和衰减）
            actual_amp = self.sample_signal(t, position) * amp_scale
            attenuated.append(actual_amp)
            
            # 记录相位
            phase_shift = self.current_theta
            phases.append(phase_shift)
            
            # 更新相位（模拟实时变化）
            if t >= self.preamble_duration:  # 仅在前导码之后更新相位
                if (t - self.preamble_duration) % (self.sweep_time/self.phase_shifts_per_sweep) == 0:
                    self.current_theta += self.phase_step
                    self.current_theta = np.clip(self.current_theta, -np.pi/2, np.pi/2)
        
        # 绘图
        plt.figure(figsize=(12, 8))
        
        # 原始信号
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, original, label='Original Signal')
        plt.title(f"AP{self.ap_id} Signal Components @ ({norm_pos[0]:.2f}, {norm_pos[1]:.2f})")
        plt.ylabel("Amplitude (V)")
        plt.legend()
        
        # 衰减后信号
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, attenuated, color='orange', label='Attenuated Signal')
        plt.ylabel("Amplitude (V)")
        plt.legend()
        
        # 相位变化
        theta_values = []
        for t in timestamps:
            if t < self.preamble_duration:
                theta_values.append(-np.pi/2)  # 前导码阶段固定角度
            else:
                # 计算扫描阶段的线性变化
                scan_progress = (t - self.preamble_duration) / (self.sweep_time - self.preamble_duration)
                theta = -np.pi/2 + scan_progress * np.pi
                theta_values.append(theta)

        # 相位变化图
        plt.subplot(3, 1, 3)
        for j in range(2, self.antenna_num+1):
            phases = [(j-1)*np.sin(theta) for theta in theta_values]  # 以π为单位
            
            # 分阶段绘制
            # 前导码阶段（灰色背景区域）
            preamble_mask = [t < self.preamble_duration for t in timestamps]
            plt.plot(np.array(timestamps)[preamble_mask], 
                    np.array(phases)[preamble_mask], 
                    color='gray', 
                    linewidth=1.5,
                    alpha=0)
            
            # 扫描阶段（彩色实线）
            scan_mask = [t >= self.preamble_duration for t in timestamps]
            plt.plot(np.array(timestamps)[scan_mask], 
                    np.array(phases)[scan_mask], 
                    linewidth=1.5,
                    label=f'Antenna {j} Phase')

        # 设置纵坐标刻度为π的倍数
        max_phase = (self.antenna_num-1)  # 最大相位值（π的倍数）
        y_ticks = np.arange(-max_phase, max_phase+1)
        plt.yticks(y_ticks, [f'{x}π' if x !=0 else '0' for x in y_ticks])
        plt.xlabel("Time (ms)")
        plt.ylabel("Phase (π rad)")
        plt.grid(alpha=0.3)
        plt.legend()

        # 添加前导码区域标注
        plt.axvspan(0, self.preamble_duration, color='gray', alpha=0.2, label='Preamble')
        
        plt.tight_layout()
        output_dir = f'sim_figures/AP_{self.ap_id}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"signal_at_position_{norm_pos[0]:.2f}_{norm_pos[1]:.2f}.png"), 
                  dpi=300, bbox_inches='tight')
        plt.close()

    def statistics(self):
        print (f"\nAccess Point {self.ap_id} Statistics:")
        print (f"├─ Location: {[round(i, 3) for i in self.location]}")
        print (f"├─ Preamble Duration: {self.preamble_duration}ms")
        print (f"├─ Carrier Frequency: {self.freq/1e6}MHz")
        print (f"├─ Single Period: {1/(self.freq)*1e9:.4f}ns")
        print (f"└─ Antenna Configuration:")
        print (f"   ├─ Number: {self.antenna_num}")
        print (f"   ├─ Spacing: {self.antenna_spacing:.4f}m")
        print (f"   └─ Direction: {self.antenna_direction:.2f} radians")

class Simulation:
    def __init__(self, food_sources):
        self.start_time = datetime.datetime.now()
        self.bees = [Bee(food_sources) for _ in range(NUM_BEES)]
        self.food_sources = food_sources
        self.frame_count = 0

    def update_nectar(self):
        self.frame_count += 1
        for _, fs in enumerate(self.food_sources):
            if fs.nectar == -1: 
                if fs.recovery_timer > 0:
                    fs.recovery_timer -= 1
                else:
                    fs.nectar = round((0.8 + 0.4 * random.random()) * NECTAR_EXPECTED_STORAGE)
                    fs.recovery_timer = 0

    def update_bees(self):
        for bee in self.bees:
            if not bee.has_food: 

                # Clear the target food if it is found to be empty
                for fs in self.food_sources:
                    distance = np.linalg.norm(fs.position - bee.position)
                    if distance < NECTAR_EMPTY_DISCOVERY_DISTANCE:
                        if fs.nectar <= 0:
                            bee.known_empty_sources.add(fs)
                        else:
                            bee.target_food = fs
                            bee.known_empty_sources.discard(fs)
                    
                    if (distance < NECTAR_DISCOVERY_DISTANCE and 
                        fs != bee.target_food and fs.nectar > 0 and 
                        fs not in bee.known_empty_sources):
                        if bee.target_food is None:
                            bee.target_food = fs
                        else:
                            current_dist = np.linalg.norm(bee.target_food.position - bee.position)
                            if distance < current_dist:
                                bee.target_food = fs
                if (bee.target_food and bee.target_food in bee.known_empty_sources):
                    bee.target_food = None
                
                # Find a new target food
                if not bee.target_food:
                    potential_sources = [
                        fs for fs in self.food_sources 
                        if fs not in bee.known_empty_sources
                    ]
                    if potential_sources:
                        distances = [np.linalg.norm(fs.position - bee.position) for fs in potential_sources]
                        bee.target_food = potential_sources[np.argmin(distances)]
                    else:
                        bee.reset_knowledge()
                
                # Move to the target food
                if bee.target_food:
                    direction = bee.target_food.position - bee.position
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        step_size = np.clip(STEP_SIZE * distance, MIN_STEP, MAX_STEP)
                        step = (direction / distance) * step_size
                        bee.position += step + np.random.normal(0, STEP_NOISE, 2)

                    # Arrived at the target food
                    if np.linalg.norm(bee.position - bee.target_food.position) < OVERLAP_DISTANCE_ERROR:
                        if bee.target_food.nectar > 0:
                            bee.target_food.nectar -= 1
                            bee.has_food = True
                        else:
                            bee.known_empty_sources.add(bee.target_food)
                        bee.target_food = None
            else:
                # Move back to Hive
                direction = np.array(HIVE_POS) - bee.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    step_size = np.clip(STEP_SIZE * distance, MIN_STEP, MAX_STEP)
                    step = (direction / distance) * step_size
                    bee.position += step + np.random.normal(0, STEP_NOISE, 2)
                
                    # Arrived at Hive
                    if np.linalg.norm(bee.position - HIVE_POS) < OVERLAP_DISTANCE_ERROR:
                        bee.has_food = False

        for _, fs in enumerate(self.food_sources):
            if fs.nectar == 0: 
                fs.nectar = -1
                fs.recovery_timer = 1000 / FRAME_TIME * NECTAR_EXPECTED_RECOVERY_TIME * (0.8 + 0.4 * random.random())

    def run_frame(self, frame):
        current_time = frame * FRAME_TIME
        
        # TDMA调度：每个AP在分配的时间窗口发送
        for i, ap in enumerate(APs):
            if current_time % (len(APs)*ap.sweep_time) == i*ap.sweep_time:
                ap.transmit_signal(self.bees, current_time)
        
        # ...保持原有运动逻辑不变...

# Generate APs [STATIC]
APs = []
for i, pos in enumerate(AP_POS):
    new_AP = AP(i, 
                  (pos[0] * FIELD_SIZE, pos[1] * FIELD_SIZE / FIELD_ASPECT_RATIO),
                  AP_NUM_ANTENNAS, 
                  AP_FREQ, 
                  AP_PREAMBLES[i], 
                  sweep_time = AP_SWEEP_T, 
                  preamble_duration = AP_SWEEP_PREAMBLE_T, 
                  phase_shifts_per_sweep = AP_PHASESHIFT_NUM)
    # All APs are coarsely synchronized using TDMA, 
    # given the same frequency, all APs uniformly distribute the AP_SWEEP_T
    # so that the signal generation is coherent
    new_AP.set_signal_start((-1 + i / len(AP_POS)) * AP_SWEEP_T)
    APs.append(new_AP)

# Generate food sources [STATIC]
def generate_valid_position(existing_positions, min_dist=0.05):
    while True:
        pos = np.random.rand(2)
        pos[0] = 0.05 + 0.9 * pos[0]
        pos[1] = 0.05 + 0.9 * pos[1]
        if all(np.linalg.norm(pos - p) >= min_dist for p in existing_positions):
            return pos
all_positions = [HIVE_POS] + AP_POS
food_sources = []
for _ in range(NUM_FOOD_SOURCES):
    pos = generate_valid_position(all_positions)
    all_positions.append(pos)
    food_sources.append(Food(pos))

# Output the environment topology
print("\nEnvironment Topology: ")
print(f"├─ Field Size: {FIELD_SIZE}x{FIELD_SIZE/FIELD_ASPECT_RATIO:.1f} meters")
print(f"├─ AP Positions (Total {len(APs)}):")
for i, ap in enumerate(APs):
    print(f"│   {'├─' if i < len(APs)-1 else '└─'} AP {i}: [{ap.location[0]:.3f}, {ap.location[1]:.3f}] m")

hive_pos_m = (
    HIVE_POS[0] * FIELD_SIZE,
    HIVE_POS[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
)
print(f"├─ Hive Position: [{hive_pos_m[0]:.3f}, {hive_pos_m[1]:.3f}] m")
print(f"└─ Food Sources (Total {len(food_sources)}):")
for i, fs in enumerate(food_sources):
    fs_pos_m = (
        fs.position[0] * FIELD_SIZE,
        fs.position[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
    )
    connector = '├─' if i < len(food_sources)-1 else '└─'
    print(f"    {connector} Source {i}: [{fs_pos_m[0]:.3f}, {fs_pos_m[1]:.3f}] m")
print ()

# Find NLOS [STATIC]
PENETRATION_LOSS_HIVE = 15
PENETRATION_LOSS_AP = 10
PENETRATION_LOSS_FOOD = 5
PATH_OBSTACLE_DISTANCE = 1  # Along the path, 1m inside is considered penetrated
OBSCATLE_ATTENUATION_FACTOR = 50
FREESPACE_ATTENUATION_FACTOR = 20
def find_nlos():
    """
    Improved NLOS attenuation calculation, integrating angle information
    Save attenuation values (dB) to sim_stats/AP_{i}/nlos_attenuation.csv
    """
    os.makedirs('sim_stats', exist_ok=True)
    
    # Get all obstacle positions (converted to meters)
    obstacles = []
    obstacles.append(('hive', hive_pos_m, PENETRATION_LOSS_HIVE))
    for ap in APs:
        obstacles.append(('ap', ap.location, PENETRATION_LOSS_AP))
    for fs in food_sources:
        fs_pos_m = (fs.position[0]*FIELD_SIZE, fs.position[1]*FIELD_SIZE/FIELD_ASPECT_RATIO)
        obstacles.append(('food', fs_pos_m, PENETRATION_LOSS_FOOD))

    # Generate attenuation map for each AP
    for ap in APs: 
        ap_dir = f'sim_stats/AP_{ap.ap_id}'
        csv_path = os.path.join(ap_dir, 'nlos_attenuation.csv')
        print (f"Generating NLOS attenuation map for AP{ap.ap_id} ...")
            
        os.makedirs(ap_dir, exist_ok=True)
        
        # Generate grid (1m resolution)
        x_coords = np.arange(0, FIELD_SIZE, 1)
        y_coords = np.arange(0, FIELD_SIZE / FIELD_ASPECT_RATIO, 1)
        
        # Get first and last grid points for debugging
        debug_points = [(x_coords[0], y_coords[0]), 
                       (x_coords[-1], y_coords[-1])]
        
        with open(csv_path, 'w') as f:
            f.write("x,y,total_attenuation_dB\n")  # 改为保存衰减值
            
            attenuation_grid = np.zeros((FIELD_SIZE, int(FIELD_SIZE/FIELD_ASPECT_RATIO)))
            
            for x in x_coords:
                for y in y_coords:
                    current_pos = np.array([x, y])
                    total_atten = 0
                    
                    # Calculate the LOS direction to the AP
                    los_vector = current_pos - ap.location
                    los_distance = np.linalg.norm(los_vector)
                    if los_distance == 0:
                        continue
                    fs_loss = np.log10(4 * np.pi * los_distance * ap.freq / 3e8) * FREESPACE_ATTENUATION_FACTOR
                    
                    # Calculate attenuation components
                    attenuation = 0.0
                    for _, obs_pos, loss in obstacles:
                        if np.array_equal(obs_pos, ap.location):
                            continue
                        
                        # 计算障碍物到AP-接收点连线的距离
                        ap_pos = ap.location
                        recv_pos = np.array([x, y])
                        vec_AP2Recv = recv_pos - ap_pos
                        vec_AP2Obs = obs_pos - ap_pos
                        
                        # Find the projection parameter t
                        t = np.dot(vec_AP2Obs, vec_AP2Recv) / np.dot(vec_AP2Recv, vec_AP2Recv)
                        vec_AP2Recv_3d = np.append(vec_AP2Recv, 0)
                        vec_AP2Obs_3d = np.append(vec_AP2Obs, 0)
                        cross_product = np.cross(vec_AP2Recv_3d, vec_AP2Obs_3d)
                        distance = np.linalg.norm(cross_product) / np.linalg.norm(vec_AP2Recv)
                        if (0 <= t <= 1) and (distance < PATH_OBSTACLE_DISTANCE):
                            attenuation += loss
                    
                    total_atten = fs_loss + attenuation
                    debug_mode = (x, y) in debug_points
                    if debug_mode:
                        print(f"\nAttenuation AP{ap.ap_id} @ ({round(x,2)},{round(y,2)})")
                        print(f"├─ LOS Vector: {los_vector}")
                        print(f"├─ LOS Distance: {los_distance:.2f}m")
                        print(f"├─ Free Space Loss: {fs_loss:.2f}dB")
                        print(f"├─ NLOS Attenuation: {attenuation:.2f}dB")
                        print(f"└─ Total Attenuation: {total_atten:.2f}dB")
                    
                    attenuation_grid[int(x), int(y)] = total_atten
                    f.write(f"{x},{y},{total_atten:.1f}\n")
            
            plot_attenuation_map(ap.ap_id, attenuation_grid)

def plot_attenuation_map(ap_id, attenuation_grid):
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    vmax = np.percentile(attenuation_grid, 95)
    img = plt.imshow(attenuation_grid.T,  # 保存imshow对象以便后续引用（可选）
                    cmap='plasma',
                    origin='lower',
                    vmin=0,
                    vmax=vmax,
                    extent=[0, FIELD_SIZE, 0, FIELD_SIZE/FIELD_ASPECT_RATIO])
    
    # 添加等高线
    X, Y = np.meshgrid(np.arange(FIELD_SIZE), np.arange(int(FIELD_SIZE/FIELD_ASPECT_RATIO)))
    levels = np.linspace(0, vmax, 15)
    plt.contour(X, Y, attenuation_grid.T, 
               levels=levels, 
               colors='white', 
               linewidths=0.3, 
               alpha=0.5)
    
    # 调整colorbar（移除update_normal）
    cbar = plt.colorbar(img, label='Total Attenuation (dB)')  # 直接关联img对象更安全
    cbar.set_ticks(np.linspace(0, vmax, 10))
    cbar.outline.set_edgecolor('black')
    plt.setp(cbar.ax.yaxis.get_ticklines(), color='black')
    
    # 标注当前AP
    ap_loc = APs[ap_id].location
    plt.scatter(ap_loc[0], ap_loc[1], 
               marker='^', s=200, c='red', 
               edgecolors='white', label=f'AP {ap_id}')
    
    # 标注其他AP
    for i, ap in enumerate(APs):
        if i != ap_id:
            plt.scatter(ap.location[0], ap.location[1], 
                       marker='^', s=120, c='red',
                       alpha=0.5)
    
    # 标注蜂巢
    plt.scatter(hive_pos_m[0], hive_pos_m[1], 
               marker='s', s=120, c='green', 
               alpha=0.5)
    
    # 标注蜜源
    for fs in food_sources:
        fs_pos = (fs.position[0]*FIELD_SIZE, 
                 fs.position[1]*FIELD_SIZE/FIELD_ASPECT_RATIO)
        plt.scatter(fs_pos[0], fs_pos[1], 
                   marker='o', s=80, c='blue',
                   alpha=0.5)
    
    plt.title(f"AP {ap_id} Signal Attenuation Map")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    
    output_dir = f'sim_figures/AP_{ap_id}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "attenuation_map.png"), 
              dpi=300, bbox_inches='tight')
    plt.close()

find_nlos()# 测试AP0的中心点




## Start Simulation
# Draw the initial elements
sim = Simulation(food_sources)
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_facecolor('black')
bee_scatter = ax.scatter(
    [], [], 
    c='yellow', 
    s=50, 
    alpha=0.2,
    edgecolors='white',
    label='Bees'
)
ax.plot([], [], 'ro', markersize=12, alpha=0.8)
ax.plot(HIVE_POS[0], HIVE_POS[1], 'gs', markersize=15, label='Hive')
food_scatter = ax.scatter(
    [fs.position[0] for fs in food_sources],
    [fs.position[1] for fs in food_sources],
    c='#1E90FF', 
    s=[fs.nectar / NECTAR_EXPECTED_STORAGE * NECTAR_DISPLAY_SIZE * 2 for fs in food_sources],
    edgecolors='white'
)
for i, ap in enumerate(AP_POS):
    label = 'AP' if i == 0 else None
    ax.plot(ap[0], ap[1], 'r^', markersize=15, alpha=0.8, label=label)
food_labels = [ax.text(fs.position[0], fs.position[1]+0.02, str(fs.nectar), 
                      ha='center', va='bottom', color='white', fontsize=8)
              for fs in food_sources]
legend_elements = [
    plt.Line2D([0], [0], 
               marker='o', 
               color='none', 
               markerfacecolor='yellow',
               markersize=10, 
               label='Bees',
               alpha=0.7,
               markeredgewidth=0), 
    plt.Line2D([0], [0],
               marker='o',
               color='none',
               markerfacecolor='#1E90FF',
               markersize=10,
               label='Food',
               markeredgewidth=0),
    plt.Line2D([0], [0],
               marker='s', 
               color='none',
               markerfacecolor='green',
               markersize=10,
               label='Hive',
               markeredgewidth=0),
    plt.Line2D([0], [0],
               marker='^', 
               color='none',
               markerfacecolor='red',
               markersize=10,
               label='AP',
               markeredgewidth=0)
]
ax.legend(handles=legend_elements, loc='upper right')

# Start the AP phase shift algorithm

# Update the frame
prev_save_time = 0
current_frame_id = 0
trail_save_num = 10
if (NUM_FRAMES < 100): 
    trail_save_num = math.ceil(NUM_FRAMES / 10)

def update(frame):
    global current_frame_id
    current_frame_id = frame

    # Update bee history and save the bee trail
    def save_bee_trails(frame, current_time):
        os.makedirs('sim_figures', exist_ok=True)
        for i, bee in enumerate(sim.bees):
            bee_dir = f'sim_figures/bee_{i:02d}'
            os.makedirs(bee_dir, exist_ok=True)
            
            # Create a new canvas
            fig_bee, ax_bee = plt.subplots(figsize=FIG_SIZE)
            ax_bee.set_xlim(0, 1)
            ax_bee.set_ylim(0, 1)
            ax_bee.set_facecolor('black')
            
            # Draw the access points
            for ap in AP_POS:
                ax_bee.plot(ap[0], ap[1], 'r^', markersize=15, alpha=0.8)
            
            # Draw the trajectory of this bee
            trail_positions = []
            trail_alphas = []
            for pos, timestamp in bee.history:
                age = current_time - timestamp
                alpha = max(0, 1 - age/MOVEMENT_DISAPPEAR_TIME)
                if alpha > 0:
                    trail_positions.append(pos)
                    trail_alphas.append(alpha)
            
            if trail_positions:
                ax_bee.scatter(
                    [p[0] for p in trail_positions],
                    [p[1] for p in trail_positions],
                    c="yellow", 
                    s=20,
                    alpha=trail_alphas,
                    edgecolors='none'
                )
            global prev_save_time
            plt.savefig(f'{bee_dir}/trail_{prev_save_time}ms-{current_time}ms.png', dpi=150, bbox_inches='tight')
            plt.close(fig_bee)
    for bee in sim.bees:
        bee.update_position_history(current_frame_id * FRAME_TIME)
    if (frame + 1) % (NUM_FRAMES // trail_save_num) == 0:
        current_time = current_frame_id * FRAME_TIME
        save_bee_trails(frame+1, current_time)  
        global prev_save_time
        prev_save_time = current_time + FRAME_TIME
    
    # Update the bees and nectars (and their displays)
    sim.update_bees()
    sim.update_nectar()
    positions = np.array([b.position for b in sim.bees])
    alphas = [0.7 if b.has_food else 0.2 for b in sim.bees]
    bee_scatter.set_offsets(positions)
    bee_scatter.set_alpha(alphas)
    food_sizes = []
    food_alphas = []
    for fs in sim.food_sources:
        if fs.nectar > 0:
            size = (1 + (fs.nectar / NECTAR_EXPECTED_STORAGE)) * NECTAR_DISPLAY_SIZE
            alpha = 0.7
        else:
            size = NECTAR_DISPLAY_SIZE
            alpha = 0.2
        food_sizes.append(size)
        food_alphas.append(alpha)
    food_scatter.set_sizes(food_sizes)
    food_scatter.set_alpha(food_alphas)
    for fs, label in zip(sim.food_sources, food_labels):
        if fs.nectar > 0: 
            label.set_text(str(fs.nectar))
            label.set_visible(True)
        else:
            label.set_visible(False)
    return [bee_scatter, food_scatter] + food_labels

# Save the animation
print("\nGenerating simulation amination ...")
ani = FuncAnimation(
    fig, 
    update, 
    frames=NUM_FRAMES, 
    interval=FRAME_TIME, 
    blit=True,
    repeat=False
)
ani.save('sim.gif', 
         writer='pillow', 
         fps=1000//FRAME_TIME,
         progress_callback=lambda i, n: print(f'Progress: {i+1}/{n} frames', end='\r'))
print("\nAnimation saved! ")
