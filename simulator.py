'''
Paper: Living IoT:AFlyingWireless Platformon LiveInsects
Bee flight simulation with a focus on the localization with multiple APs
'''
import numpy as np
import os, datetime, math, shutil, random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import threading  # 添加线程支持
import time  # 添加时间模块
import gc  # 添加垃圾回收支持
import sys  # 添加sys模块

if os.path.exists('sim.gif'):
    os.remove('sim.gif')
if os.path.exists('sim_figures'):
    shutil.rmtree('sim_figures')


# Environment and Display Parameters
NUM_BEES = 2
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
AP_SWEEP_T = 50
AP_SWEEP_PREAMBLE_T = 5   # time (ms) of preamble in a sweep <- self chosen
AP_PHASESHIFT_NUM = 30  # 减少相位变化次数
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
SIM_TIME = 100   # ms
NUM_FRAMES = round(SIM_TIME / FRAME_TIME)

# Bee Parameters
# bumblebee movement speed: x m/s (normally)
# which is (x / FIELD_SIZE) * normalized_edge_length m/s
# 1s = 20 frames, so 1 frame = (x / FIELD_SIZE / 20) * normalized_edge_length m
FIELD_SIZE = 100         # m. Distance between two APs is 50, close to the paper configuration
FIELD_ASPECT_RATIO = 1.0 # width / height
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
AP_SAMPLE_RATE = 4 * AP_FREQ  # 4倍过采样（足够捕获信号特征）
AP_SIGNAL_AMPLITUDE = 10**(AP_SIGNAL_STRENGTH / 20)  # 转换dBm为线性振幅

# Debug print lock
debug_print_lock = threading.Lock()

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
        self.signal_memory = {}  # 结构: {ap_id: [(timestamp, sample), ...]}
        self.plotted_aps = set()  # 记录已绘制过的AP

    def reset_knowledge(self):
        self.known_empty_sources.clear()

    def update_position_history(self, frame_time):
        self.history.append((self.position.copy(), frame_time))
        if len(self.history) > math.ceil(MOVEMENT_DISAPPEAR_TIME / FRAME_TIME):
            self.history.pop(0)

    def receive_signal(self, ap_id, timestamps, samples):
        '''存储接收到的信号数据，保留最近3个sweep周期的数据'''
        # 获取当前AP的sweep周期
        ap = next((a for a in APs if a.ap_id == ap_id), None)
        if not ap: return

        # 合并新数据
        new_data = list(zip(timestamps, samples))
        if ap_id not in self.signal_memory:
            self.signal_memory[ap_id] = []
        
        # 合并并排序
        self.signal_memory[ap_id].extend(new_data)
        # 仅在需要时排序（比如绘图前）

        # 首次达到3T时绘制信号
        if ap_id not in self.plotted_aps:
            if len(self.signal_memory.get(ap_id, [])) >= 3 * APs[ap_id].sweep_time / (1000/AP_SAMPLE_RATE):
                self.plot_ap_signals(ap_id)
                self.plotted_aps.add(ap_id)

        # 输出接收日志
        if timestamps:
            print (f"\nBee {id(self)} 接收AP{ap_id}信号")
            print (f"├─ 首时间戳: {timestamps[0]:.2f}ms")
            print (f"├─ 末时间戳: {timestamps[-1]:.2f}ms")
            duration = timestamps[-1] - timestamps[0]
            print (f"├─ 持续时间: {duration:.2f}ms")
            print (f"└─ 对应周期数: {duration / ap.sweep_time:.1f}T")

        # 添加严格的内存限制
        max_samples = 3 * ap.sweep_time * AP_SAMPLE_RATE / 1000  # 精确计算3T样本数
        if len(self.signal_memory[ap_id]) > max_samples * 1.2:  # 允许20%冗余
            self.signal_memory[ap_id] = self.signal_memory[ap_id][-int(max_samples):]

    def plot_ap_signals(self, ap_id):
        '''绘制指定AP的完整信号波形'''
        if ap_id not in self.signal_memory:
            return
        
        data = self.signal_memory[ap_id]
        
        # Use the first timestamp and the last timestamp to decide whether
        # the signal duration has achieved 3T
        if not data or (data[-1][0] - data[0][0]) < 3 * APs[ap_id].sweep_time:
            return
        
        timestamps, samples = zip(*data)
        
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, samples, linewidth=0.5, alpha=0.7)
        plt.title(f"Bee {id(self)} Received Signals from AP{ap_id}\n"
                 f"Duration: {max(timestamps)-min(timestamps):.1f}ms")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (V)")
        
        # 标注关键区域
        ap = next(a for a in APs if a.ap_id == ap_id)
        for t in np.arange(min(timestamps), max(timestamps), ap.sweep_time):
            plt.axvspan(t, t+ap.preamble_duration, color='gray', alpha=0.2)
        
        output_dir = f'sim_figures/bee_{id(self)}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"AP_{ap_id}_3Tsignal.png"), 
                  dpi=200, bbox_inches='tight')
        plt.close()

        print (f"Bee {id(self)} 已为AP{ap_id}生成信号图")
        print (f"├─ 样本数: {len(data)}")
        print (f"└─ 时间范围: {max(t)-min(t):.1f}ms")

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

        # self.transmit_lock = threading.Lock()
        self.test_transmit_time = 0 # testing purpose
        self.test_transmit_count = 0 # testing purpose

        # Preload NLOS attenuation data and base signal template
        self.attenuation_map = None
        t = np.linspace(0, 1/self.freq, int(AP_SAMPLE_RATE/self.freq))
        self.base_signal = np.sin(2 * np.pi * self.freq * t)
        
        # Performance counters
        self.preamble_sample_count = 0
        self.preamble_sample_time = 0.0
        self.scan_sample_count = 0
        self.scan_sample_time = 0.0

    def load_attenuation_map(self):
        '''
        Load attenuation map when initializing
        '''
        attenuation_map = {}
        nlos_attenuation_path = os.path.join(os.path.dirname(__file__),"sim_stats", f"AP_{self.ap_id}", 'nlos_attenuation.csv')
        if (not os.path.exists(nlos_attenuation_path)):
            raise FileNotFoundError(f"NLOS attenuation map file not found: {nlos_attenuation_path}")
        with open(nlos_attenuation_path, 'r') as f:
            for line in f:
                x_str, y_str, atten = line.strip().split(',')
                # 先转浮点数再转整数，处理可能的.0后缀
                x = int(float(x_str))
                y = int(float(y_str))
                attenuation_map[(x, y)] = float(atten)
        self.attenuation_map = attenuation_map

    def calculate_beamforming_phase(self, position):
        '''
        Calculate beamforming phase difference based on the paper formula
        '''
        # Calculate the arrival angle θ (relative to the antenna array direction)
        vec = position - self.location
        theta = np.arctan2(vec[1], vec[0]) - self.antenna_direction
        theta = np.clip(theta, -np.pi/2, np.pi/2)
        
        # 根据论文公式设置相位（天线编号从2开始）
        for j in range(2, self.antenna_num+1):
            self.antenna_phases[j-1] = (j-1) * np.pi * np.sin(theta)
        
        return sum(self.antenna_phases[1:])  # 合成相位（忽略0号天线）

    def calculate_attenuation(self, position):
        '''优化后的衰减计算'''
        x = int(position[0] * FIELD_SIZE)
        y = int(position[1] * FIELD_SIZE / FIELD_ASPECT_RATIO)
        return self.attenuation_map.get((x, y), 0.0)
    
    def sample_signal(self, timestamp, position=None):
        '''
        Generate a signal with beamforming effect
        Signal composition: Global phase offset + beamforming phase difference
                 ┌───────────────┐
        Time ──→ │ Phase shifter │──→ Actual signal
                 └─────┬─────┬───┘
                    Global  Beamforming phase
                    shift   difference
        '''
        if (self.signal_start is None or timestamp < self.signal_start):
            return 0 # signal generation not started yet
        
        # Global phase offset (due to TDMA scheduling)
        phase_shift = np.radians(self.current_theta)
        if position is not None:
            delta_phase = self.calculate_beamforming_phase(position)
            phase_shift += delta_phase

        equivalent_signal_start = self.signal_start + self.sweep_time * ((timestamp - self.signal_start) // self.sweep_time)
        if (timestamp < equivalent_signal_start + self.preamble_duration): 
            # Preamble phase
            return self.preamble[int((timestamp - equivalent_signal_start) / self.preamble_duration * len(self.preamble))]
        else: 
            # Scan phase (Sweep phase)
            time = (timestamp - equivalent_signal_start - self.preamble_duration)
            return np.sin(2 * np.pi * self.freq * (time * 1e-3) + phase_shift)

    def transmit_signal(self, bees, transmit_start_time):
        start_time = time.perf_counter()
        
        print (f"\nAP{self.ap_id} 开始传输 @ {transmit_start_time}ms")
        print (f"├─ 目标蜜蜂数: {len(bees)}")
        print (f"├─ 信号时长: {self.sweep_time}ms")
        print (f"└─ 采样率: {AP_SAMPLE_RATE/1e6:.2f}MHz")

        if not bees:
            print ("  → 无目标蜜蜂")
            return

        # 重置波束角度
        self.current_theta = -np.pi/2
        self.signal_start = transmit_start_time  # 设置信号起始时间

        # 生成完整信号时间轴
        timestamps = np.arange(
            transmit_start_time,
            transmit_start_time + self.sweep_time,
            1000 / AP_SAMPLE_RATE
        )
        
        # 为每个蜜蜂生成定制化信号
        for bee in bees:
            if not hasattr(bee, 'position'):
                print (f"  → 无效蜜蜂对象: {id(bee)}")
                continue
            
            # 计算该蜜蜂的衰减系数
            attenuation = self.calculate_attenuation(bee.position)
            amp_scale = 10**(-attenuation/20)
            
            # 生成针对该蜜蜂的信号
            samples = [
                self.sample_signal(t, bee.position) * amp_scale
                for t in timestamps
            ]
            
            print (f"  → 发送给 Bee{id(bee)}")
            print (f"    ├─ 样本数: {len(samples)}")
            print (f"    └─ 时间范围: {timestamps[0]:.2f} - {timestamps[-1]:.2f}ms")
            
            bee.receive_signal(self.ap_id, timestamps, samples)

        duration = (time.perf_counter() - start_time) * 1000
        print (f"  AP{self.ap_id} 传输耗时: {duration:.2f}ms")

    def signal_characteristics(self):
        '''
        Generate dual-view signal analysis with separated time scales
        '''
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
        plt.savefig(os.path.join(output_dir, "signal_characteristics.png"), 
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

    def signal_charactertistics_at_position(self, norm_pos, save_fig=False):
        '''
        Verify the signal characteristics at a specific position
        '''
        position = np.array(norm_pos)
        timestamps = np.arange(0, self.sweep_time, 0.1)  # 0.1ms resolution
        original = []
        attenuated = []
        phases = []

        for t in timestamps:

            # Original signal (no attenuation)
            amp = self.sample_signal(t, position)
            original.append(amp)
            
            # Actual received signal (with beamforming and attenuation)
            attenuation = self.calculate_attenuation(position)
            amp_scale = AP_SIGNAL_AMPLITUDE * 10**(-attenuation/20)
            actual_amp = self.sample_signal(t, position) * amp_scale
            attenuated.append(actual_amp)
            
            # Simulate the phase shift
            phase_shift = self.current_theta
            phases.append(phase_shift)
            if t >= self.preamble_duration:  # Only update phase after preamble
                if (t - self.preamble_duration) % (self.sweep_time/self.phase_shifts_per_sweep) == 0:
                    self.current_theta += self.phase_step
                    self.current_theta = np.clip(self.current_theta, -np.pi/2, np.pi/2)
        
        if (not save_fig): 
            pass
        

        # Original signal figure
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, original, label='Original Signal')
        plt.title(f"AP{self.ap_id} Signal Components @ ({norm_pos[0]:.2f}, {norm_pos[1]:.2f})")
        plt.ylabel("Amplitude (V)")
        plt.legend()
        
        # Attenuated signal figure
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, attenuated, color='orange', label='Attenuated Signal')
        plt.ylabel("Amplitude (V)")
        plt.legend()
        
        theta_values = []
        for t in timestamps:
            if t < self.preamble_duration: 
                # Preamble phase fixed theta value [the preamble line will not be displayed]
                theta_values.append(-np.pi/2)
            else:
                # Calculate the theta value linear change in the scan phase
                scan_progress = (t - self.preamble_duration) / (self.sweep_time - self.preamble_duration)
                theta = -np.pi/2 + scan_progress * np.pi
                theta_values.append(theta)

        # Phase change figure
        plt.subplot(3, 1, 3)
        for j in range(2, self.antenna_num+1):
            phases = [(j-1)*np.sin(theta) for theta in theta_values]  # 以π为单位
            
            # Preamble phase [the preamble line will not be displayed]
            # preamble_mask = [t < self.preamble_duration for t in timestamps]
            # plt.plot(np.array(timestamps)[preamble_mask], 
            #         np.array(phases)[preamble_mask], 
            #         color='gray', 
            #         linewidth=1.5,
            #         alpha=0)
            
            # Scan phase (Sweep phase)
            scan_mask = [t >= self.preamble_duration for t in timestamps]
            plt.plot(np.array(timestamps)[scan_mask], 
                    np.array(phases)[scan_mask], 
                    linewidth=1.5,
                    label=f'Antenna {j} Phase')

        # Set y-axis ticks as multiples of π
        max_phase = (self.antenna_num-1)
        y_ticks = np.arange(-max_phase, max_phase+1)
        plt.yticks(y_ticks, [f'{x}π' if x !=0 else '0' for x in y_ticks])
        plt.xlabel("Time (ms)")
        plt.ylabel("Phase (π rad)")
        plt.grid(alpha=0.3)
        plt.legend()

        # Add preamble region annotation
        plt.axvspan(0, self.preamble_duration, color='gray', alpha=0.2, label='Preamble')
        plt.tight_layout()
        output_dir = f'sim_figures/AP_{self.ap_id}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"signal_charactertistics_at_position_{norm_pos[0]:.2f}_{norm_pos[1]:.2f}.png"), 
                  dpi=300, bbox_inches='tight')
        plt.close()

    def get_signal_start(self):
        return self.signal_start

    def set_signal_start(self, timestamp):
        self.signal_start = timestamp # in ms

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
print ("\nEnvironment Topology: ")
print (f"├─ Field Size: {FIELD_SIZE}x{FIELD_SIZE/FIELD_ASPECT_RATIO:.1f} meters")
print (f"├─ AP Positions (Total {len(APs)}):")
for i, ap in enumerate(APs):
    print (f"│   {'├─' if i < len(APs)-1 else '└─'} AP {i}: [{ap.location[0]:.3f}, {ap.location[1]:.3f}] m")

hive_pos_m = (
    HIVE_POS[0] * FIELD_SIZE,
    HIVE_POS[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
)
print (f"├─ Hive Position: [{hive_pos_m[0]:.3f}, {hive_pos_m[1]:.3f}] m")
print (f"└─ Food Sources (Total {len(food_sources)}):")
for i, fs in enumerate(food_sources):
    fs_pos_m = (
        fs.position[0] * FIELD_SIZE,
        fs.position[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
    )
    connector = '├─' if i < len(food_sources)-1 else '└─'
    print (f"    {connector} Source {i}: [{fs_pos_m[0]:.3f}, {fs_pos_m[1]:.3f}] m")
print ()

# Find NLOS [STATIC]
PENETRATION_LOSS_HIVE = 15
PENETRATION_LOSS_AP = 10
PENETRATION_LOSS_FOOD = 5
PATH_OBSTACLE_DISTANCE = 1  # Along the path, 1m inside is considered penetrated
OBSCATLE_ATTENUATION_FACTOR = 50
FREESPACE_ATTENUATION_FACTOR = 20

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

def find_nlos():
    '''
    NLOS attenuation calculation only consider free space loss and obstacle attenuation
    '''
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

            # Format: x,y,total_attenuation_dB
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
                        print (f"\nAttenuation AP{ap.ap_id} @ ({round(x,2)},{round(y,2)})")
                        print (f"├─ LOS Vector: {los_vector}")
                        print (f"├─ LOS Distance: {los_distance:.2f}m")
                        print (f"├─ Free Space Loss: {fs_loss:.2f}dB")
                        print (f"├─ NLOS Attenuation: {attenuation:.2f}dB")
                        print (f"└─ Total Attenuation: {total_atten:.2f}dB")
                    
                    attenuation_grid[int(x), int(y)] = total_atten
                    f.write(f"{int(round(x))},{int(round(y))},{total_atten:.1f}\n")
            
            plot_attenuation_map(ap.ap_id, attenuation_grid)
find_nlos()

for ap in APs: 
    nlos_attenuation_path = os.path.join(os.path.dirname(__file__),"sim_stats", f"AP_{ap.ap_id}", 'nlos_attenuation.csv')
    if (not os.path.exists(nlos_attenuation_path)):
        find_nlos()
    ap.load_attenuation_map()

# Run the tests
for ap in APs:
    ap.signal_characteristics()
    ap.signal_charactertistics_at_position((0.0, 0.0))

exit(1)  # stop here



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



# Update the frame
current_frame_id = 0
trail_save_num = 10
prev_save_time = 0
if (NUM_FRAMES < 100): 
    trail_save_num = math.ceil(NUM_FRAMES / 10)

# Last sample time of each AP
# Note that the AP start time is not 0
last_AP_sample_time = [ap.get_signal_start() for ap in APs]
frame_0_outputed = False
def update(frame):
    global current_frame_id, frame_0_outputed
    frame_start = time.perf_counter()
    current_frame_id = frame
    current_time = current_frame_id * FRAME_TIME

    # 传输信号
    if (not frame_0_outputed): print (f"\nFrame {frame} 开始处理  |  当前时间: {current_time}ms")
    tx_start = time.perf_counter()
    transmit_threads = []
    for ap in APs:
        while current_time - last_AP_sample_time[ap.ap_id] > ap.sweep_time:
            thread = threading.Thread(
                target=ap.transmit_signal,
                args=(sim.bees, last_AP_sample_time[ap.ap_id]),
                daemon=True
            )
            transmit_threads.append(thread)
            thread.start()
            last_AP_sample_time[ap.ap_id] += ap.sweep_time
    for thread in transmit_threads:
        thread.join(timeout=0.01)
    
    tx_duration = (time.perf_counter() - tx_start) * 1000
    if (not frame_0_outputed): print (f"信号传输总耗时: {tx_duration:.2f}ms")

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

    # 渲染部分
    render_start = time.perf_counter()
    # ...保持渲染代码...
    render_duration = (time.perf_counter() - render_start) * 1000
    if (not frame_0_outputed): print (f"渲染耗时: {render_duration:.2f}ms")

    total_duration = (time.perf_counter() - frame_start) * 1000
    if (not frame_0_outputed): print (f"Frame {frame} 总耗时: {total_duration:.2f}ms\n")
    if (frame == 0): frame_0_outputed = True
    return [bee_scatter, food_scatter] + food_labels

# Save the animation
print ("\nGenerating simulation amination ...")
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
         progress_callback=lambda i, n: print (f'Progress: {i+1}/{n} frames', end='\r'))
print ("\nAnimation saved! ")
plt.close('all')
os._exit(0)  # 强制终止进程
