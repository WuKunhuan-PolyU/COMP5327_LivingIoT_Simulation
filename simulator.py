'''
Paper: Living IoT:AFlyingWireless Platformon LiveInsects
Bee flight simulation with a focus on the localization with multiple APs
'''
import numpy as np
import os, datetime, math, shutil, random, time, threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

if os.path.exists('sim.gif'):
    os.remove('sim.gif')
if os.path.exists('sim_figures'):
    shutil.rmtree('sim_figures')


# Environment and Display Parameters
NUM_BEES = 2
NUM_FOOD_SOURCES = 5  # randomly distributed in the field
HIVE_POS_NORM = (0.5, 0.1)                  # normalized
AP_POS_NORM = [(0.5, 0.0), (0.0, 0.5)]    # normalized
AP_NUM = len(AP_POS_NORM)
FIG_SIZE = (8, 8)
NECTAR_DISPLAY_SIZE = 50
NECTAR_EXPECTED_STORAGE = 15
NECTAR_EXPECTED_RECOVERY_TIME = 20

# AP Parameters
AP_PREAMBLES = [(1,0,1,0,1,0,1,0), (1,1,0,0,1,1,0,0)]
# numbers are chosen for the ease of simulation
AP_SWEEP_T = 50
AP_PREAMBLE_T = 5   # time (ms) of preamble in a sweep <- self chosen
AP_PHASESHIFT_NUM = 50
assert (NUM_BEES * AP_NUM * AP_PHASESHIFT_NUM <= 10000)
# in (50, 5, 90) setting, each phase shift takes 0.5ms
# as TDMA is used, the signals are transmitted in a time-division manner
# for example, if there are 2 APs, and the sweep time is 50ms
# the first 25ms, only AP0 is transmitting, 
# the second 25ms, only AP1 is transmitting
AP_NUM_ANTENNAS = 2
AP_FREQ = 915e6         # 915 MHz
AP_SIGNAL_STRENGTH = 28 # dBm
AP_SIGNAL_AMPLITUDE = 10**(AP_SIGNAL_STRENGTH / 20)  # 转换dBm为线性振幅
AP_ANTENNA_LAYOUT_DIRECTION = [0, 3 * np.pi / 2] # radians, counterclockwise from the positive x-axis

# Simulation Parameters
FRAME_TIME = 50    # ms
SIM_TIME = 500     # ms
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

# Debug print lock
debug_print_lock = threading.Lock()

# Class definitions
class Food:
    def __init__(self, location):
        self.location_normalized = location
        self.nectar = round((0.8 + 0.4 * random.random()) * NECTAR_EXPECTED_STORAGE)
        self.recovery_timer = 0

class Bee:
    def __init__(self, bee_id, food_sources):
        self.bee_id = bee_id
        self.location_normalized = np.array(HIVE_POS_NORM)
        self.target_food = None
        self.food_sources = food_sources
        self.has_food = False
        self.known_empty_sources = set() # Local memory of empty food sources
        self.history = []  # Record location and time
        self.signal_memory = [] # Record the signal memory. Format: [(timestamp, amplitude), ...]
        self.signal_memory_preambles = [] # Record preamble signal's timestamp. Format: [((start_timestamp, ap_id), sampled_timestamp_amps), ...]
        self.signal_memory_ap_angles = []  # Record the AP angle from max amplitude timestamps. Format: [((max_amp_timestamp, ap_id), angle), ...]
        self.signal_received_count = 0

        # testing purposes
        self.signal_memory_debug_count = 0
        self.signal_receive_plotted = False

        # 前导码参数预计算
        self.bit_duration = AP_PREAMBLE_T / len(AP_PREAMBLES[0])  # 每个比特持续时间
        self.preamble_states = {
            'idle': 0,
            'detecting': 1,
            'verifying': 2
        }
        self.current_preamble_state = self.preamble_states['idle']
        self.detected_bits = []
        self.detection_start_time = None

        # Quick-access variables
        self.sweep_interval = AP_SWEEP_T * AP_NUM

    def reset_knowledge(self):
        self.known_empty_sources.clear()

    def update_location_history(self, frame_time):
        self.history.append((self.location_normalized.copy(), frame_time))
        if len(self.history) > math.ceil(MOVEMENT_DISAPPEAR_TIME / FRAME_TIME):
            self.history.pop(0)
    
    def detect_preamble_signal(self, samples):
        '''
        基于时间窗口和相对振幅的轻量级前导码检测
        '''
        bit_duration = AP_PREAMBLE_T / len(AP_PREAMBLES[0])
        min_stable_time = 0.3 * bit_duration  # 至少稳定0.3ms视为有效比特
        
        # 第一步：检测可能的起始比特
        stable_start = None
        current_max = None
        for i in range(len(samples)-1):
            current_ts, current_amp = samples[i]
            next_ts, _ = samples[i+1]
            if stable_start is None:
                stable_start = current_ts
                current_max = current_amp
            else: 
                current_max = max(current_max, current_amp)
                rel_amp = current_amp / current_max
                if (current_ts - stable_start) >= min_stable_time: 
                    sampled_timestamp_amps = []
                    sampled_bits = []
                    start_time = stable_start

                    # Estimate the sampling gap of preamble signals
                    expected_samples_per_bit = bit_duration / (next_ts - current_ts)
                    if i + int((len(AP_PREAMBLES[0])-1) * expected_samples_per_bit) >= len(samples):
                        print (f"sample length is not enough: {len(samples)}, expected at least {i + int((len(AP_PREAMBLES[0])-1) * expected_samples_per_bit)}")
                        return None # sample length is not enough
                    
                    # Verify the detected bits
                    for bit_idx in range(len(AP_PREAMBLES[0])): 
                        sampled_bits.append(samples[i + int(bit_idx * expected_samples_per_bit)][1])
                        sampled_timestamp_amps.append(samples[i + int(bit_idx * expected_samples_per_bit)])
                    try: 
                        detected_bits = np.array(sampled_bits)
                        detected_min = np.min(detected_bits)
                        detected_bits -= detected_min
                        detected_range = np.max(detected_bits) # note that min is 0 now
                        detected_bits /= detected_range
                        detected_bits = np.where((-0.1 < detected_bits) & (detected_bits < 0.1), 0, detected_bits)
                        detected_bits = np.where((0.9 < detected_bits) & (detected_bits < 1.1), 1, detected_bits)
                        detected_bits = tuple(detected_bits)
                        ap_id = AP_PREAMBLES.index(detected_bits) # will throw an error if not found
                        # Find the exact ending sample id for the preamble
                        end_idx = i + int((len(AP_PREAMBLES[0])-1) * expected_samples_per_bit)
                        while (end_idx < len(samples)): 
                            current_bit = samples[end_idx][1]
                            normalized_last_preamble_bit = (sampled_bits[-1] - detected_min) / detected_range
                            normalized_current_bit = (current_bit - detected_min) / detected_range
                            if (normalized_last_preamble_bit != normalized_current_bit):
                                break
                            end_idx += 1 # find the exact ending sample id for the preamble
                        if self.signal_memory_debug_count <= 3:
                            print(f"\nBee {self.bee_id} detected preamble @ {stable_start:.2f}ms")
                            print(f"├─ AP ID: {ap_id}")
                            print(f"└─ Sampled bits: {np.round(np.array(sampled_bits), 5)}")
                        return (start_time, ap_id), sampled_timestamp_amps, end_idx
                    except Exception as _: 
                        return None # detected bits are not valid
                elif abs(rel_amp - 1.0) >= 0.1: 
                    stable_start = None
                    current_max = None
        return None

    def _reset_preamble_state(self):
        self.current_preamble_state = self.preamble_states['idle']
        self.detected_bits = []
        self.detection_start_time = None

    def find_angle_from_max_amplitude(self, signal_segment, ap_id):
        # 找到最大幅度对应的相位索引
        max_idx = np.argmax([s[1] for s in signal_segment])
        max_time = signal_segment[max_idx][0]
        
        # 通过预存参数计算相位
        sweep_start = (max_time // self.sweep_interval) * self.sweep_interval
        phase_index = int(
            (max_time - sweep_start - AP_PREAMBLE_T) / 
            (AP_SWEEP_T - AP_PREAMBLE_T) * AP_PHASESHIFT_NUM
        ) % AP_PHASESHIFT_NUM
        
        # 根据相位索引计算角度
        theta = -np.pi/2 + np.pi * phase_index / AP_PHASESHIFT_NUM
        wavelength = 3e8 / AP_FREQ
        phase_diff = theta * wavelength / (2 * np.pi)
        ratio = phase_diff / (0.5 * wavelength)
        
        # 处理浮点误差导致的数值溢出
        safe_ratio = np.clip(ratio, -1.0, 1.0)
        angle = np.arcsin(safe_ratio)
        return (max_time, ap_id), np.degrees(angle), signal_segment[max_idx][1]

    def plot_received_signal(self, bee_id, signal_data, mark=False):
        '''
        Plot the received signal from all APs
        '''
        if not signal_data: return
        
        plt.figure(figsize=(12, 6))
        times = [s[0] for s in signal_data]
        amps = [s[1] for s in signal_data]
        
        # Plot the raw signal
        plt.plot(times, amps, 'b-', alpha=0.3, label='Raw Signal')
        if (mark): 
            # Mark the preamble region and the sampling points
            plotted_preambles = []
            preamble_sample_labelled = False
            labelled_angle_aps = []
            for ts_and_ap_id, sampled_timestamp_amps in self.signal_memory_preambles:
                start = sampled_timestamp_amps[0][0]
                end = sampled_timestamp_amps[-1][0]
                if ts_and_ap_id[1] not in plotted_preambles:
                    plt.axvspan(start, end, alpha=0.2, color=['red','blue'][ts_and_ap_id[1]], label=f'AP{ts_and_ap_id[1]} Preamble')
                    plotted_preambles.append(ts_and_ap_id[1])
                else: 
                    plt.axvspan(start, end, alpha=0.2, color=['red','blue'][ts_and_ap_id[1]])
                if not preamble_sample_labelled:
                    plt.scatter([sample[0] for sample in sampled_timestamp_amps], 
                                [sample[1] for sample in sampled_timestamp_amps], 
                                c='red', s=30, zorder=3, 
                                label='Preamble Samples')
                    preamble_sample_labelled = True
                else: 
                    plt.scatter([sample[0] for sample in sampled_timestamp_amps], 
                                [sample[1] for sample in sampled_timestamp_amps], 
                                c='red', s=30, zorder=3)
            
            # Mark the angle calculation result
            rblues = ['#008888', '#007777', '#006666', '#005555', '#004444', '#003333']
            for (max_amp_ts, ap_id), angle, point_amp in self.signal_memory_ap_angles: 
                if ap_id not in labelled_angle_aps:
                    plt.scatter(max_amp_ts, point_amp, c=rblues[len(labelled_angle_aps)%len(rblues)], s=60, zorder=3,
                                label=f'AP{ap_id} Angle = {angle:.1f}°')
                    labelled_angle_aps.append(ap_id)
                else: 
                    plt.scatter(max_amp_ts, point_amp, c=rblues[len(labelled_angle_aps)%len(rblues)], s=60, zorder=3)
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.title(f'Bee {bee_id} Received Signal')
        plt.legend()
        

        os.makedirs(f'sim_figures/bee_{bee_id:02d}', exist_ok=True)
        plt.savefig(f'sim_figures/bee_{bee_id:02d}/received_signal{"_raw_" + str(self.signal_memory_debug_count) if not mark else ""}.png', 
                      dpi=200, bbox_inches='tight')
        plt.close()

    def receive_signal(self, timestamps, samples):
        '''
        Store the received signal data, keep the last 3 sweep periods
        '''
        new_data = list(zip(timestamps, samples))
        self.signal_memory_debug_count += 1
        
        # format: timestamp,amplitude
        if (len(new_data) > 0): 
            if (self.signal_memory_debug_count <= 3): 
                self.data_dir = f"sim_stats/bee_{self.bee_id:02d}"
                os.makedirs(self.data_dir, exist_ok=True)
                with open(f"{self.data_dir}/received_signal_raw_{self.signal_memory_debug_count}.csv", "w") as f:
                    for t, s in zip(timestamps, samples):
                        f.write(f"{t:.5f},{s:.5f}\n")
                self.plot_received_signal(self.bee_id, new_data)

        # Merge and check whether sorted
        # i.e., if the signal memory is not empty, 
        # verify whether the new data is sorted, otherwise discard the observation
        if (len(self.signal_memory) > 0): 
            if (new_data[0][0] < self.signal_memory[-1][0]):
                print (f"Bee {id(self)} received signal ({new_data[0][0]:.2f}ms - {new_data[-1][0]:.2f}ms): Unsorted timestamps (latest received: {self.signal_memory[-1][0]:.2f}ms)")
                return

        self.signal_memory.extend(new_data)
        before_crop_len = len(self.signal_memory)



        # 处理信号内存裁剪
        preserve_duration = (AP_NUM + 1) * AP_SWEEP_T
        first_preserved_timestamp = self.signal_memory[-1][0] - preserve_duration if self.signal_memory else None
        
        # Prior-knowledge guess to cut the first len(new_data) samples
        # General method is to use binary search to find the first timestamp
        if (len(self.signal_memory) > 0 and 
            self.signal_memory[0][0] < first_preserved_timestamp): 
            if (len(self.signal_memory) > len(new_data) and 
                self.signal_memory[len(new_data) - 1][0] < first_preserved_timestamp and 
                self.signal_memory[len(new_data)][0] >= first_preserved_timestamp):
                self.signal_memory = self.signal_memory[len(new_data):]
                if (self.signal_memory_debug_count < 3): print  (f"CROP the memory using prior knowledge (start from {len(new_data)})")
            else:
                left, right = len(new_data), len(self.signal_memory)
                while left < right:
                    mid = (left + right) // 2
                    if (first_preserved_timestamp is not None) and (self.signal_memory[mid][0] < first_preserved_timestamp):
                        left = mid + 1
                    else:
                        right = mid
                self.signal_memory = self.signal_memory[left:] if left < len(self.signal_memory) else []
                if (self.signal_memory_debug_count < 3): print  (f"CROP the memory using binary search (start from {left})")

        # Cut the preambles and angles memory at the same time
        # Not too costly, as each sweep period => only 1 statistics saved, 
        # so just perform linear search
        self.signal_memory_preambles = [p for p in self.signal_memory_preambles if p[0][0] >= first_preserved_timestamp]
        self.signal_memory_ap_angles = [a for a in self.signal_memory_ap_angles if a[0][0] >= first_preserved_timestamp]

        # Search for new preambles in the new data
        # "locate preamble2 in S_(i+T) .. S_(3T) at i" indicates that preamble pairs should 
        # have at least T time gap between them
        # According to the paper, detection can start from the last preamble end even recorded
        # the new data must be preserved (not cropped due to exceeding the time limit)
        assert (len(self.signal_memory) >= len(new_data)), f"len(self.signal_memory): {len(self.signal_memory)}, len(new_data): {len(new_data)}"
        preamble_search_index = len(self.signal_memory) - len(new_data)
        preamble_detection = None
        while (self.signal_memory_preambles and preamble_search_index < len(self.signal_memory) and
               self.signal_memory[preamble_search_index][0] < self.signal_memory_preambles[-1][0][0] + AP_SWEEP_T): 
            preamble_search_index += 1
        if (preamble_search_index < len(self.signal_memory)):
            result = self.detect_preamble_signal(self.signal_memory[preamble_search_index:])
            if (result):
                preamble_info, sampled_timestamp_amps, end_idx = result
                preamble_detection = preamble_info[1]
                self.signal_memory_preambles.append((preamble_info, sampled_timestamp_amps))
                
                # 精确提取相位扫描信号段
                signal_start_idx = preamble_search_index + end_idx
                sweep_end_time = preamble_info[0] + AP_SWEEP_T
                signal_segment = []
                for i in range(signal_start_idx, len(self.signal_memory)):
                    ts, amp = self.signal_memory[i]
                    if ts > sweep_end_time:
                        break
                    signal_segment.append( (ts, amp) )
                
                # 计算角度
                max_amp_sample, angle, point_amp = self.find_angle_from_max_amplitude(signal_segment, preamble_detection)
                if angle:
                    self.signal_memory_ap_angles.append((max_amp_sample, angle, point_amp))
        # Output the signal receive information
        if (self.signal_memory_debug_count < 30 and 
            self.signal_memory_debug_count % 10 == 9 and 
            len(timestamps) > 0): 
            print (f"\nBee {self.bee_id} received signal")
            print (f"├─ First timestamp: {timestamps[0]:.2f}ms")
            print (f"├─ Last timestamp: {timestamps[-1]:.2f}ms")
            duration = timestamps[-1] - timestamps[0]
            print (f"├─ Duration: {duration:.2f}ms")
            print (f"├─ Num of new samples: {len(new_data)}")
            print (f"├─ Num of deleted samples: {before_crop_len - len(self.signal_memory)}")
            print (f"├─ Num of stored samples: {len(self.signal_memory)}")
            if (preamble_detection): 
                print (f"├─ AP detected: {preamble_detection}")
            print (f"└─ ap_id: (max_amp_ts, angle): \n{[f'{ts_and_ap_id[1]}: {round(ts_and_ap_id[0], 2)}ms, {round(angle, 2)}°' for ts_and_ap_id, angle, _ in self.signal_memory_ap_angles]}")
            self.plot_received_signal(self.bee_id, self.signal_memory, mark = True)
            self.signal_receive_plotted = True

    def sample_location_history(self):
        pass
    
    def get_estimated_location(self):
        pass

class AP:
    def __init__(self, ap_id, 
                 location, # (x (in m), y (in m))
                 antenna_num, # assume the antennas are distributed in a line, and the center is the location
                 frequency, # in Hz
                 preamble_signal, 
                 antenna_layout_direction = 0, # radians, counterclockwise from the positive x-axis
                 ):
        self.ap_id = ap_id
        self.location = np.array(location) # not normalized, in meters
        self.location_normalized = self.location / FIELD_SIZE
        self.preamble = preamble_signal

        # used to calculate the signal
        self.signal_start = 0  # in ms
        self.freq = frequency  # in Hz

        self.antenna_num = antenna_num
        self.antenna_layout_direction = antenna_layout_direction
        self.wavelength = 3e8 / frequency  # Speed of light / frequency, in meters
        
        # CRITICAL: 天线布局方向向量
        # According to the paper, say there are 2 antennas, 
        # When the first antenna has a higher distance to the target, namely Φ, 
        # There will be a relationship: arccos(Φ / d) = θ, and θ is within [0, π/2]
        self.antenna_spacing = self.wavelength / 2  # Half wavelength spacing, in meters, should be around 33cm for 915MHz
        antenna_layout_direction_vector = np.array([np.cos(antenna_layout_direction), np.sin(antenna_layout_direction)])
        self.antenna_locations = [
            self.location + self.antenna_spacing * (i - (self.antenna_num-1)/2) * antenna_layout_direction_vector
            for i in range(self.antenna_num)
        ]

        # 波束成形相关参数  
        self.phase_step = np.pi / AP_PHASESHIFT_NUM  # δ = π/90 ≈ 2°
        self.antenna_phases = [0.0] * (self.antenna_num + 1)  # 天线编号从1开始

        self.test_transmit_time = 0 # testing purpose
        self.test_transmit_count = 0 # testing purpose

        # Preload NLOS attenuation data and base signal template
        self.attenuation_map = None
        
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

    def calculate_beamforming_phase(self, norm_pos):
        pos_m = norm_pos * FIELD_SIZE
        distances = [np.linalg.norm(pos_m - ant) for ant in self.antenna_locations]
        # Find Φ for each antenna
        # Every wavelength is equivalent to 2π phase shift
        phase_diffs = [(d - distances[0]) * (2 * np.pi / self.wavelength) for d in distances[1:]]
        # if (norm_pos[0] == 0.5 and norm_pos[1] == 0.5):
        #     print  (f"\n========= calculate beamforming phase =========")
        #     print  (f"norm_pos: {norm_pos}, pos_m: {pos_m}")
        #     print  (f"distances: {[round(i, 2) for i in distances]}")
        #     print  (f"phase_diffs: {[round(i, 2) for i in phase_diffs]}")
        return np.array(phase_diffs)

    def calculate_attenuation(self, location):
        '''
        Retrieve the attenuation from the attenuation map
        '''
        x = int(location[0] * FIELD_SIZE)
        y = int(location[1] * FIELD_SIZE / FIELD_ASPECT_RATIO)
        return self.attenuation_map.get((x, y), 0.0)
    
    def sample_signal(self, timestamp, location=None): 
        '''
        Generate a signal with beamforming effect given the timestamp
        Signal comlocation: Global phase offset + beamforming phase difference
                 ┌───────────────┐
        Time ──→ │ Phase shifter │──→ Actual signal
                 └─────┬─────┬───┘
                    Global  Beamforming phase
                    shift   difference
        '''
        # signal generation not started yet, OR
        # the current time is not designed for this AP to transmit
        if (self.signal_start is None or timestamp < self.signal_start or 
            (timestamp - self.signal_start) % (AP_NUM * AP_SWEEP_T) >= AP_SWEEP_T): 
            return 0
        
        # Global phase offset (due to TDMA scheduling)
        # Calculate the beamformed phase shift <- all the antennas
        # This only applies when the location is given
        beam_phases = [0 for _ in range(self.antenna_num)]
        if location is not None: 
            # Find Φ for each antenna
            beam_phases = self.calculate_beamforming_phase(location)
        equivalent_signal_start = self.signal_start + AP_SWEEP_T * ((timestamp - self.signal_start) // AP_SWEEP_T)
        if (timestamp < equivalent_signal_start + AP_PREAMBLE_T): 
            # Preamble phase
            return self.preamble[int((timestamp - equivalent_signal_start) / AP_PREAMBLE_T * len(self.preamble))]
        else: 
            # Scan phase (Sweep phase)
            # normalize the signal amplitude from all antennas
            # phase_shift_by_AP is the θ component in y(t) = |ax(t) + ae^(j(ϕ−θ)) * x(t)|
            sweep_timestamp = timestamp - equivalent_signal_start - AP_PREAMBLE_T
            phase_shift_by_AP = -np.pi/2 + (np.pi / AP_PHASESHIFT_NUM) * int(AP_PHASESHIFT_NUM * (sweep_timestamp - AP_PREAMBLE_T) / (AP_SWEEP_T - AP_PREAMBLE_T))
            return sum([np.sin(2 * np.pi * self.freq * (sweep_timestamp * 1e-3) + phase_shift_by_AP + phase) for phase in beam_phases]) / len(beam_phases)

    def transmit_signal(self, bees, timestamp):
        '''
        每个sweep周期生成AP_PHASESHIFT_NUM个样本
        '''
        # 检查是否在传输窗口
        if (self.signal_start is None or 
            timestamp < self.signal_start or 
            (timestamp - self.signal_start) % (AP_NUM * AP_SWEEP_T) >= AP_SWEEP_T):
            return 0
        
        # 生成该sweep周期的时间轴
        sweep_start = self.signal_start + ((timestamp - self.signal_start) // AP_SWEEP_T) * AP_SWEEP_T
        timestamps, raw_samples = self.generate_signal(sweep_start)
        
        # 发送给所有蜜蜂
        for bee in bees:
            attenuation = self.calculate_attenuation(bee.location_normalized)
            amp_scale = 10**(-attenuation/20)
            samples = [s * amp_scale for s in raw_samples]  # 现在s是单个数值
            bee.receive_signal(timestamps, samples)
        
        return 1

    def generate_signal(self, sweep_start):
        '''
        修复为返回（时间轴，样本列表）
        '''
        # 分阶段生成时间轴
        preamble_samples = 10 * len(AP_PREAMBLES[0])
        scan_samples = AP_PHASESHIFT_NUM
        
        # 生成时间轴
        preamble_ts = np.linspace(sweep_start, sweep_start+AP_PREAMBLE_T, preamble_samples, endpoint=False)
        scan_ts = np.linspace(sweep_start+AP_PREAMBLE_T, sweep_start+AP_SWEEP_T, scan_samples, endpoint=False)
        timestamps = np.concatenate([preamble_ts, scan_ts]).tolist()
        
        # 生成每个时间点的信号
        samples = []
        for t in timestamps:
            if t < sweep_start + AP_PREAMBLE_T:
                # 前导码方波生成
                symbol_idx = int((t - sweep_start) / AP_PREAMBLE_T * len(AP_PREAMBLES[self.ap_id]))
                samples.append(AP_SIGNAL_AMPLITUDE * AP_PREAMBLES[self.ap_id][symbol_idx % len(AP_PREAMBLES[self.ap_id])])
            else:
                # 相位扫描信号生成
                phase_index = int((t - sweep_start - AP_PREAMBLE_T)/(AP_SWEEP_T - AP_PREAMBLE_T)*AP_PHASESHIFT_NUM)
                phase_shift = -np.pi/2 + (2*np.pi/AP_PHASESHIFT_NUM)*phase_index
                samples.append(np.sin(phase_shift))  # 简化波束成形计算
        
        return timestamps, samples

    def get_signal_start(self):
        return self.signal_start

    def set_signal_start(self, 
                         timestamp # in ms
                         ):
        self.signal_start = timestamp

    def test_signal_characteristics(self, timestamp):
        '''
        Generate preamble and actual signal analysis with separated time scales
        The AP time-dependent phase shift is considered in the figure
        '''

        # Find the signal period
        period_ns = 1e9 / self.freq  # in ns
        preamble_duration_ms = AP_PREAMBLE_T  # in ms
        num_periods = 3
        signal_duration_ns = (num_periods+1) * period_ns  # in ns, one more period allow phase shift part finish drawing
        
        # Dynamic sampling settings
        samples_preamble = 100  # total preamble samples
        samples_per_period = 1000  # samples per period
        
        # Generate test signal data
        preamble_t = np.linspace(timestamp, timestamp + preamble_duration_ms, samples_preamble, endpoint=False)
        signal_t = np.linspace(timestamp + preamble_duration_ms, 
                             timestamp + preamble_duration_ms + signal_duration_ns * 1e-6,  # in ns to ms
                             (num_periods+1) * samples_per_period, endpoint=False)
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
        ax1.set_title(f"Preamble Digital Signal ({preamble_duration_ms}ms)\n"
                      f"(Record start t: {timestamp}ms)")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Digital Level")
        ax1.grid(True, axis='y', linestyle=':')
        ax1.set_xlim(timestamp - 0.1, timestamp + preamble_duration_ms + 0.1)
        ax1.set_ylim(-0.1, 1.1)
        
        # Signal period view (ns scale)
        ax2 = plt.subplot(gs[1])
        relative_t = (signal_t - timestamp - preamble_duration_ms) * 1e6  # in ns
        equivalent_signal_start = self.signal_start + AP_SWEEP_T * (
            (timestamp - self.signal_start) // AP_SWEEP_T
        )
        
        # Calculate time in sweep phase (after preamble)
        # Convert phase shift to time offset (phase_shift = 2πfΔt => Δt = phase_shift/(2πf))
        time_in_sweep = timestamp - equivalent_signal_start - AP_PREAMBLE_T
        phase_progress = time_in_sweep / (AP_SWEEP_T - AP_PREAMBLE_T)
        phase_shift = -np.pi/2 + (np.pi / AP_PHASESHIFT_NUM) * int(
            AP_PHASESHIFT_NUM * phase_progress)

        time_offset_ns = (phase_shift / (2 * np.pi * self.freq)) * 1e9
        ax2.plot(relative_t, signal_data, 'b-', alpha=0.8)  # Apply time offset
        
        # Draw vertical lines at correct period boundaries
        for i in range(3):
            t = (i+1)*period_ns - time_offset_ns
            ax2.axvline(t, color='green', linestyle='--', alpha=0.6)
            ax2.text(t-period_ns/2, 0.9, 
                    f'Cycle {i+1}\n({period_ns:.2f}ns)', 
                    ha='center', fontsize=9)
        
        # Add phase shift mask and legend
        ax2.axvspan(0, -1 * time_offset_ns, 
                   facecolor='lightblue', 
                   alpha=0.3,
                   label=f'Phase Shift: {phase_shift/np.pi:.2f}π = {time_offset_ns:.2f}ns')
        ax2.legend(loc='upper right', framealpha=0.9)

        # Set axis limits relative to actual signal
        ax2.set_xlim(-period_ns*0.1, signal_duration_ns*1.1)

        # Create the overall chart
        ax2.set_title(f"Actual Signal({self.freq/1e6}MHz) (AP start t: {timestamp}ms)\n"
                      f"(Record start t: {timestamp}ms + {preamble_duration_ms}ms (preamble) <= 0ns in the chart)")
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, which='both', alpha=0.4)
        ax2.set_xlim(- period_ns * 0.1, signal_duration_ns + period_ns * 0.1 - time_offset_ns)
        ax2.set_ylim(-1.1, 1.1)
        output_dir = f'sim_figures/AP_{self.ap_id}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "signal_characteristics.png"), 
                  dpi=300, bbox_inches='tight')
        plt.close()
        print (f"\nAccess Point {self.ap_id} Statistics:")
        print (f"├─ Location: {[round(i, 3) for i in self.location]}")
        print (f"├─ Preamble Duration: {preamble_duration_ms}ms")
        print (f"├─ Signal (sweep) Duration: {signal_duration_ns:.2f}ns")
        print (f"├─ Phaseshifts per sweep: {AP_PHASESHIFT_NUM}")
        print (f"├─ Carrier Frequency: {self.freq/1e6}MHz")
        print (f"├─ Single Period: {period_ns:.4f}ns")
        print (f"├─ Wavelength: {self.wavelength:.4f}m")
        print (f"├─ Antenna Spacing: {self.antenna_spacing:.4f}m (half wavelength)")
        print (f"├─ Antenna Layout Direction: {self.antenna_layout_direction}rad")
        for i in range(self.antenna_num):
            print (f"│   {'├─' if i < self.antenna_num - 1 else '└─'} Antenna {i} Location: {[round(i, 3) for i in self.antenna_locations[i]]}")
        print (f"└─ 3 Cycles Duration: {signal_duration_ns:.2f}ns")

    def test_signal_charactertistics_at_location(self, norm_pos, save_fig=False):
        '''
        Verify the signal characteristics at a specific location
        '''
        location = np.array(norm_pos)
        timestamps = np.arange(0, AP_SWEEP_T, 0.1)  # 0.1ms resolution
        original = []
        attenuated = []
        phases = []

        for t in timestamps:

            # Original signal (no attenuation)
            amp = self.sample_signal(t, location)
            original.append(amp)
            
            # Actual received signal (with beamforming and attenuation)
            # The signal outputs for every antenna are ampliied to AP_SIGNAL_AMPLITUDE
            equivalent_signal_start = self.signal_start + AP_SWEEP_T * ((t - self.signal_start) // AP_SWEEP_T)
            attenuation = self.calculate_attenuation(location)
            amp_scale = 10 ** (-(AP_SIGNAL_AMPLITUDE - attenuation)/20)
            actual_amp = self.sample_signal(t, location) * amp_scale
            attenuated.append(actual_amp)
            # print  (AP_SIGNAL_AMPLITUDE, attenuation, amp_scale, actual_amp)
            
            # Simulate the phase shift
            phase_shift = -np.pi/2 + (np.pi / AP_PHASESHIFT_NUM) * int(AP_PHASESHIFT_NUM * (t - equivalent_signal_start - AP_PREAMBLE_T) / (AP_SWEEP_T - AP_PREAMBLE_T))
            phases.append(phase_shift)
        
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
            if t < AP_PREAMBLE_T: 
                # Preamble phase fixed theta value [the preamble line will not be displayed]
                theta_values.append(-np.pi/2)
            else:
                # Calculate the theta value linear change in the scan phase
                scan_progress = (t - AP_PREAMBLE_T) / (AP_SWEEP_T - AP_PREAMBLE_T)
                theta = -np.pi/2 + scan_progress * np.pi
                theta_values.append(theta)

        # Phase change figure
        plt.subplot(3, 1, 3)
        for j in range(2, self.antenna_num+1):
            phases = [(j-1) * np.pi * np.sin(theta) for theta in theta_values]  # 以π为单位
            
            # Preamble phase [the preamble line will not be displayed]
            # preamble_mask = [t < AP_PREAMBLE_T for t in timestamps]
            # plt.plot(np.array(timestamps)[preamble_mask], 
            #         np.array(phases)[preamble_mask], 
            #         color='gray', 
            #         linewidth=1.5,
            #         alpha=0)
            
            # Scan phase (Sweep phase)
            scan_mask = [t >= AP_PREAMBLE_T for t in timestamps]
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

        # Add preamble region annotation
        plt.axvspan(0, AP_PREAMBLE_T, color='gray', alpha=0.2, label='Preamble stage')
        plt.legend()
        plt.tight_layout()
        output_dir = f'sim_figures/AP_{self.ap_id}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"signal_charactertistics_at_location_{norm_pos[0]:.2f}_{norm_pos[1]:.2f}.png"), 
                  dpi=300, bbox_inches='tight')
        plt.close()

    def optimal_phase_to_angle(self, 
                               optimal_theta, # in radians, unit: π
                               ):
        """
        Calculate theoretical angle using paper's formula
        Discrete sampling errors and noises will cause deviation from the theoretical value
        """
        if (optimal_theta > 1): 
            target_phase_diff = 2 - optimal_theta
        else: 
            target_phase_diff = -1 * optimal_theta
        target_phase_diff_normalized = target_phase_diff
        theta_cal = np.degrees(np.arccos(-1 * target_phase_diff_normalized))
        return theta_cal
    
    def test_beamforming_optimal_phases(self):
        """
        Generate the beamforming maximum amplitude mapping to phase shifts
        """
        # Create the test grid
        grid_size = 201 # handle the edge case
        x = np.linspace(0, FIELD_SIZE, grid_size, endpoint=True)
        y = np.linspace(0, FIELD_SIZE, grid_size, endpoint=True)
        
        # 初始化存储矩阵
        max_amplitudes = np.zeros((grid_size, grid_size))
        best_phases = np.zeros((grid_size, grid_size))
        
        # 预计算相位差（调整为N-1维）
        phase_diffs = np.zeros((grid_size, grid_size, self.antenna_num-1))
        for i in range(grid_size):
            for j in range(grid_size):
                norm_pos = np.array([x[i]/FIELD_SIZE, y[j]/FIELD_SIZE])
                phase_diffs[i,j] = self.calculate_beamforming_phase(norm_pos)

        # 遍历θ范围（0到2π）
        theta_values = np.linspace(0, 2*np.pi, AP_PHASESHIFT_NUM, endpoint=False)
        for theta in theta_values:
            # 复数信号叠加计算
            complex_sum = np.ones((grid_size, grid_size), dtype=complex)  # 参考天线贡献
            for k in range(self.antenna_num-1):
                phase = theta + phase_diffs[:,:,k]  # θ补偿相位差
                complex_sum += np.exp(1j * phase)   # 复数叠加
            amplitudes = np.abs(complex_sum)  # 计算模长
            
            # 更新最大值
            mask = amplitudes > max_amplitudes
            max_amplitudes[mask] = amplitudes[mask]
            best_phases[mask] = theta % (2*np.pi)  # 相位归一化

        # Dynamic test point generation
        max_attempts = 10
        test_points = []
        for _ in range(max_attempts): 
            theta = np.random.uniform(0, 2*np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)])
            
            # Calculate AP center position (normalized)
            ap_center = self.location / FIELD_SIZE
            max_step = min(
                (1 - ap_center[0])/direction[0] if direction[0]>0 else ap_center[0]/-direction[0],
                (1 - ap_center[1])/direction[1] if direction[1]>0 else ap_center[1]/-direction[1],
                key=abs
            ) if not np.allclose(direction, 0) else 0
            if max_step < 0.2: continue
            
            # Generate two test points (distance ≥ 0.2)
            step1 = np.random.uniform(0.1, max_step-0.1)
            step2 = step1 + 0.2 + np.random.uniform(0, 0.1)
            if step2 > max_step: continue
            p1 = ap_center + direction * step1
            p2 = ap_center + direction * step2
            test_points = [p1, p2]
            break
        else: # fallback to the default test points
            test_points = [
                np.array([0.5, 0.25]) # 150 degrees: [0.5, 0.644]
            ]
            # test_points = [
            #     np.array([self.location_normalized[0] + 0.1, self.location_normalized[1]]),
            #     np.array([self.location_normalized[0] + 0.2, self.location_normalized[1]])
            # ]

        # 存储测试点信息
        test_info = []
        for point_idx, norm_pos in enumerate(test_points):
            print(f"\nTestpoint {point_idx+1} optimal phase calculation:")
            print(f"├─ Normalized coordinates: ({norm_pos[0]:.3f}, {norm_pos[1]:.3f})")
            pos_m = norm_pos * FIELD_SIZE
            print(f"├─ Actual coordinates: ({pos_m[0]:.1f}m, {pos_m[1]:.1f}m)")
            
            # Calculate the distance to each antenna
            print("├─ Antenna distance to test point:")
            for i, ant_loc in enumerate(self.antenna_locations):
                dist = np.linalg.norm(pos_m - ant_loc)
                print(f"│   {'→' if i==0 else ' '} Antenna {i}: {dist:.2f}m" + 
                     f"{' (referenced distance)' if i==0 else ''}")
            
            # Calculate the phase differences
            phase_diffs = self.calculate_beamforming_phase(norm_pos)
            print(f"├─ Antenna phase differences: {[round(pd/np.pi, 3) for pd in phase_diffs]}π")
            
            # Calculate the optimal θ and signal amplitude
            max_amp = 0
            best_theta = 0
            for theta in np.linspace(0, 2*np.pi, 50, endpoint=False):
                complex_sum = 1.0
                for pd in phase_diffs:
                    complex_sum += np.exp(1j*(theta + pd))
                amp = abs(complex_sum)
                if amp > max_amp:
                    max_amp = amp
                    best_theta = theta
            
            # Calculate the attenuation
            atten = self.calculate_attenuation(norm_pos)
            print(f"└─ Optimal θ: {best_theta/np.pi:.3f}π")
            print(f"   ├─ Maximum amplitude: {max_amp:.3f}")
            print(f"   └─ Total attenuation: {atten:.2f}dB")
            
            # 存储计算结果
            test_info.append({
                'position': norm_pos,
                'phase_diffs': [round(float(pd/np.pi),3) for pd in phase_diffs],  # 转换为原生float
                'best_theta': round(float(best_theta/np.pi),3),
                'max_amp': round(float(max_amp),3), 
                'atten': round(float(atten),2)
            })

        # 绘制热力图
        plt.figure(figsize=(12, 10))
        
        # 创建自定义颜色映射（包含alpha通道）
        blues = plt.cm.Blues(np.linspace(0.1, 0.9, 256 + 1))  # 调整颜色范围
        blues[:, 3] = np.linspace(0.1, 0.6, 256 + 1)  # 设置透明度渐变
        cmap = ListedColormap(blues)

        # 动态计算绘图范围
        img = plt.imshow(best_phases.T,  # 转置矩阵以匹配坐标方向
                  extent=[0, FIELD_SIZE, 0, FIELD_SIZE / FIELD_ASPECT_RATIO],
                  origin='lower',
                  cmap=cmap,
                  aspect='equal',
                  vmin=0, 
                  vmax=2*np.pi,
                  interpolation='bilinear')
        
        # Configure the color bar
        cbar = plt.colorbar(img, 
                           label='Optimal Phase Shift θ (rad)',
                           ticks=np.linspace(0, 2*np.pi, 5, endpoint=True),
                           extend='both')
        cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        cbar.outline.set_edgecolor('black')

        # Add environment markers
        self.draw_plot_environment_markers()
        
        # 添加天线方向箭头和角度标注
        ax = plt.gca()
        ap_x, ap_y = self.location
        ax.plot(ap_x, ap_y, 'r^', markersize=15, zorder=5)
        
        # 绘制天线阵列方向
        ant_dir = self.antenna_locations[1] - self.antenna_locations[0]
        ant_dir /= np.linalg.norm(ant_dir)
        ax.arrow(ap_x, ap_y, ant_dir[0] * 0.05 * FIELD_SIZE, ant_dir[1] * 0.05 * FIELD_SIZE, 
                head_width=2, head_length=3, fc='black', ec='black', zorder=5, 
                label='Antenna direction')
        
        # 强制设置坐标轴范围
        ax.set_xlim(0, FIELD_SIZE)
        ax.set_ylim(0, FIELD_SIZE / FIELD_ASPECT_RATIO)
        
        # 绘制测试点向量和角度标注
        for test_point in test_info:
            px, py = test_point['position'] * FIELD_SIZE
            dx, dy = px - ap_x, py - ap_y
            dx, dy = dx / np.linalg.norm([dx, dy]) * 0.1 * FIELD_SIZE, dy / np.linalg.norm([dx, dy]) * 0.1 * FIELD_SIZE
            # The actual direction
            # ax.arrow(ap_x, ap_y, dx, dy, head_width=2, head_length=3, fc='black', ec='black', linestyle='--')
            
            # 计算理论角度（考虑天线方向）
            dx_rot = dx * np.cos(self.antenna_layout_direction) + dy * np.sin(self.antenna_layout_direction)
            dy_rot = -dx * np.sin(self.antenna_layout_direction) + dy * np.cos(self.antenna_layout_direction)
            theta_geo = np.degrees(np.arctan2(dy_rot, dx_rot))
            theta_cal = self.optimal_phase_to_angle(test_point['best_theta'])

            # 绘制点
            x = test_point['position'][0] * FIELD_SIZE
            y = test_point['position'][1] * FIELD_SIZE
            plt.scatter(x, y, s=120, c='white', edgecolors='red', linewidths=1.5, zorder=4)
            va = 'bottom' if y < FIELD_SIZE*0.8 else 'top'
            plt.text(x, y+3, 
                     f"θ*={test_point['best_theta']}π, ΔΦ={test_point['phase_diffs']}π\nθ_geo={theta_geo:.1f}°, θ_cal={theta_cal:.1f}° OR {-1 * theta_cal:.1f}°", 
                    color='white', fontsize=9,
                    ha='center', va=va, 
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            # two possible theta_cal directions
            for i in [(1, '#ff0000'), (-1, '#bb0000')]: 
                dx = FIELD_SIZE * 0.2 * np.cos(theta_cal * i[0] * np.pi / 180)
                dy = FIELD_SIZE * 0.2 * np.sin(theta_cal * i[0] * np.pi / 180)
                ax.arrow(ap_x, ap_y, dx, dy, 
                        head_width=2, head_length=3, 
                        fc=i[1], ec=i[1], 
                        linestyle='--', 
                        label=f'θ_cal = {i[0] * theta_cal:.1f}°')
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels.append(l)
                unique_handles.append(h)
        ax.legend(handles=unique_handles, labels=unique_labels, 
                 loc='upper right', fontsize=8)
        
        plt.title(f"AP{self.ap_id} Optimal Phase Map\n(Phase shifts per sweep: {AP_PHASESHIFT_NUM})")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        
        # Save the figure
        save_path = os.path.join(os.path.dirname(__file__), f"sim_figures/AP_{self.ap_id}/beamforming_optimal_phases.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def draw_plot_environment_markers(self):
        """
        Unified environment markers plot
        """
        
        # Current AP
        ap_loc = to_meters(self.location_normalized)
        plt.scatter(ap_loc[0], ap_loc[1], 
                   marker='^', s=200, c='red',
                   edgecolors='white', label=f'AP {self.ap_id}')
        
        # Other APs
        for ap in APs:
            if ap.ap_id != self.ap_id:
                other_loc = to_meters(ap.location_normalized)
                plt.scatter(other_loc[0], other_loc[1],
                           marker='^', s=120, c='red',
                           alpha=0.5)
        
        # Hive
        hive_pos = to_meters(HIVE_POS_NORM)
        plt.scatter(hive_pos[0], hive_pos[1],
                   marker='s', s=120, c='green',
                   alpha=0.5)
        
        # Food sources
        for fs in food_sources:
            fs_pos = to_meters(fs.location_normalized)
            plt.scatter(fs_pos[0], fs_pos[1],
                       marker='o', s=80, c='blue',
                       alpha=0.5)
        
        # Add legend
        plt.legend(loc='upper right', framealpha=0.9)

    def get_phase_at_time(self, timestamp):
        '''
        获取指定时间点的相位配置
        '''
        sweep_index = int((timestamp - self.signal_start) // AP_SWEEP_T)
        theta_values = np.linspace(0, 2*np.pi, AP_PHASESHIFT_NUM, endpoint=False)
        return theta_values[sweep_index % len(theta_values)]

class Simulation:
    def __init__(self, food_sources):
        self.start_time = datetime.datetime.now()
        self.bees = [Bee(i, food_sources) for i in range(NUM_BEES)]
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
                    distance = np.linalg.norm(fs.location_normalized - bee.location_normalized)
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
                            current_dist = np.linalg.norm(bee.target_food.location_normalized - bee.location_normalized)
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
                        distances = [np.linalg.norm(fs.location_normalized - bee.location_normalized) for fs in potential_sources]
                        bee.target_food = potential_sources[np.argmin(distances)]
                    else:
                        bee.reset_knowledge()
                
                # Move to the target food
                if bee.target_food:
                    direction = bee.target_food.location_normalized - bee.location_normalized
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        step_size = np.clip(STEP_SIZE * distance, MIN_STEP, MAX_STEP)
                        step = (direction / distance) * step_size
                        bee.location_normalized += step + np.random.normal(0, STEP_NOISE, 2)

                    # Arrived at the target food
                    if np.linalg.norm(bee.location_normalized - bee.target_food.location_normalized) < OVERLAP_DISTANCE_ERROR:
                        if bee.target_food.nectar > 0:
                            bee.target_food.nectar -= 1
                            bee.has_food = True
                        else:
                            bee.known_empty_sources.add(bee.target_food)
                        bee.target_food = None
            else:
                # Move back to Hive
                direction = np.array(HIVE_POS_NORM) - bee.location_normalized
                distance = np.linalg.norm(direction)
                if distance > 0:
                    step_size = np.clip(STEP_SIZE * distance, MIN_STEP, MAX_STEP)
                    step = (direction / distance) * step_size
                    bee.location_normalized += step + np.random.normal(0, STEP_NOISE, 2)
                
                    # Arrived at Hive
                    if np.linalg.norm(bee.location_normalized - HIVE_POS_NORM) < OVERLAP_DISTANCE_ERROR:
                        bee.has_food = False

        for _, fs in enumerate(self.food_sources):
            if fs.nectar == 0: 
                fs.nectar = -1
                fs.recovery_timer = 1000 / FRAME_TIME * NECTAR_EXPECTED_RECOVERY_TIME * (0.8 + 0.4 * random.random())



# Convert normalized coordinates to actual meters
def to_meters(norm_pos):
    return [norm_pos[0] * FIELD_SIZE, norm_pos[1] * FIELD_SIZE / FIELD_ASPECT_RATIO]

# Generate APs
APs = []
for i, pos in enumerate(AP_POS_NORM):
    new_AP = AP(i, 
                  (pos[0] * FIELD_SIZE, pos[1] * FIELD_SIZE / FIELD_ASPECT_RATIO),
                  AP_NUM_ANTENNAS, 
                  AP_FREQ, 
                  AP_PREAMBLES[i], 
                  AP_ANTENNA_LAYOUT_DIRECTION[i])
    # All APs are coarsely synchronized using TDMA, 
    # given the same frequency, all APs uniformly distribute the AP_SWEEP_T
    # so that the signal generation is coherent
    new_AP.set_signal_start(-i * AP_SWEEP_T)
    APs.append(new_AP)

# Generate food sources
def generate_valid_location(existing_locations, min_dist=0.05):
    while True:
        pos = np.random.rand(2)
        pos[0] = 0.05 + 0.9 * pos[0]
        pos[1] = 0.05 + 0.9 * pos[1]
        if all(np.linalg.norm(pos - p) >= min_dist for p in existing_locations):
            return pos
all_locations = [HIVE_POS_NORM] + AP_POS_NORM
food_sources = []
for _ in range(NUM_FOOD_SOURCES):
    pos = generate_valid_location(all_locations)
    all_locations.append(pos)
    food_sources.append(Food(pos))

# Output the environment topology
print ("\nEnvironment Topology: ")
print (f"├─ Field Size: {FIELD_SIZE}x{FIELD_SIZE/FIELD_ASPECT_RATIO:.1f} meters")
print (f"├─ AP Positions (Total {AP_NUM}):")
for i, ap in enumerate(APs):
    print (f"│   {'├─' if i < AP_NUM-1 else '└─'} AP {i}: [{ap.location[0]:.3f}, {ap.location[1]:.3f}] m")

hive_pos_m = (
    HIVE_POS_NORM[0] * FIELD_SIZE,
    HIVE_POS_NORM[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
)
print (f"├─ Hive Position: [{hive_pos_m[0]:.3f}, {hive_pos_m[1]:.3f}] m")
print (f"└─ Food Sources (Total {len(food_sources)}):")
for i, fs in enumerate(food_sources):
    fs_pos_m = (
        fs.location_normalized[0] * FIELD_SIZE,
        fs.location_normalized[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
    )
    connector = '├─' if i < len(food_sources)-1 else '└─'
    print (f"    {connector} Source {i}: [{fs_pos_m[0]:.3f}, {fs_pos_m[1]:.3f}] m")

# Find NLOS [STATIC]
PENETRATION_LOSS_HIVE = 15
PENETRATION_LOSS_AP = 10
PENETRATION_LOSS_FOOD = 5
PATH_OBSTACLE_DISTANCE = 1  # Along the path, 1m inside is considered penetrated
OBSCATLE_ATTENUATION_FACTOR = 50
FREESPACE_ATTENUATION_FACTOR = 20

# Find NLOS attenuation map
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
    levels = np.linspace(0, vmax, 15 + 1)
    plt.contour(X, Y, attenuation_grid.T, 
               levels=levels, 
               colors='white', 
               linewidths=0.3, 
               alpha=0.5)
    
    # 调整colorbar
    cbar = plt.colorbar(img, label='Total Attenuation (dB)')
    cbar.set_ticks(np.linspace(0, vmax, 10 + 1))
    cbar.outline.set_edgecolor('black')
    plt.setp(cbar.ax.yaxis.get_ticklines(), color='black')

    # Create environment markers
    APs[ap_id].draw_plot_environment_markers()

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
    
    # Get all obstacle locations (converted to meters)
    obstacles = []
    obstacles.append(('hive', hive_pos_m, PENETRATION_LOSS_HIVE))
    for ap in APs:
        obstacles.append(('ap', ap.location, PENETRATION_LOSS_AP))
    for fs in food_sources:
        fs_pos_m = (fs.location_normalized[0]*FIELD_SIZE, fs.location_normalized[1]*FIELD_SIZE/FIELD_ASPECT_RATIO)
        obstacles.append(('food', fs_pos_m, PENETRATION_LOSS_FOOD))

    # Generate attenuation map for each AP
    for ap in APs: 
        ap_dir = f'sim_stats/AP_{ap.ap_id}'
        csv_path = os.path.join(ap_dir, 'nlos_attenuation.csv')
        print (f"\nGenerating NLOS attenuation map for AP{ap.ap_id} ...")
            
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
                    fs_loss = np.log10(4 * np.pi * los_distance / ap.wavelength) * FREESPACE_ATTENUATION_FACTOR
                    
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

# Run the AP tests
for ap in APs: 
    print (f"\nTest AP{ap.ap_id} Signal Characteristics ...")
    ap.test_signal_characteristics(ap.signal_start)
    print (f"\nTest AP{ap.ap_id} Signal Characteristics at (0,0) ...")
    ap.test_signal_charactertistics_at_location((0.0, 0.0))
    print (f"\nTest AP{ap.ap_id} Beamforming Optimal Phases ...")
    ap.test_beamforming_optimal_phases()



exit(1)  # 测试结束

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
ax.plot(HIVE_POS_NORM[0], HIVE_POS_NORM[1], 'gs', markersize=15, label='Hive')
food_scatter = ax.scatter(
    [fs.location_normalized[0] for fs in food_sources],
    [fs.location_normalized[1] for fs in food_sources],
    c='#1E90FF', 
    s=[fs.nectar / NECTAR_EXPECTED_STORAGE * NECTAR_DISPLAY_SIZE * 2 for fs in food_sources],
    edgecolors='white'
)
for i, ap in enumerate(AP_POS_NORM):
    label = 'AP' if i == 0 else None
    ax.plot(ap[0], ap[1], 'r^', markersize=15, alpha=0.8, label=label)
food_labels = [ax.text(fs.location_normalized[0], fs.location_normalized[1]+0.02, str(fs.nectar), 
                      ha='center', va='bottom', color='white', fontsize=8)
              for fs in food_sources]
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='yellow', markersize=10, 
               label='Bees', alpha=0.7, markeredgewidth=0), 
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#1E90FF', markersize=10,
               label='Food', markeredgewidth=0),
    plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='green', markersize=10,
               label='Hive', markeredgewidth=0),
    plt.Line2D([0], [0], marker='^', color='none', markerfacecolor='red', markersize=10,
               label='AP', markeredgewidth=0)
]
ax.legend(handles=legend_elements, loc='upper right')



# Determine how many trail traces to save
current_frame_id = 0
trail_save_num = 10
prev_save_time = 0
if (NUM_FRAMES < 100): 
    trail_save_num = math.ceil(NUM_FRAMES / 10)

# Last sample time of each AP
last_AP_sample_time = [ap.get_signal_start() for ap in APs]
frame_0_outputed = False

# Update the frame
def update(frame):
    global current_frame_id, frame_0_outputed
    frame_start = time.perf_counter()
    current_frame_id = frame
    current_time = current_frame_id * FRAME_TIME

    # Transmit signals
    if (not frame_0_outputed): print (f"\nFrame {frame} starts processing  |  Current time: {current_time}ms")
    transmitted = True
    while transmitted: 
        transmitted = False
        for ap in APs: 
            while current_time - last_AP_sample_time[ap.ap_id] > AP_SWEEP_T: 
                tx_start = time.perf_counter()
                ap.transmit_signal(sim.bees, last_AP_sample_time[ap.ap_id])
                last_AP_sample_time[ap.ap_id] += AP_SWEEP_T * AP_NUM
                tx_duration = (time.perf_counter() - tx_start) * 1000
                if (not frame_0_outputed): print (f"AP{ap.ap_id} signal ({round(last_AP_sample_time[ap.ap_id], 1)}ms) transmission: {tx_duration:.2f}ms")
                transmitted = True

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
            for ap in AP_POS_NORM:
                ax_bee.plot(ap[0], ap[1], 'r^', markersize=15, alpha=0.8)
            
            # Draw the trajectory of this bee
            trail_locations = []
            trail_alphas = []
            for pos, timestamp in bee.history:
                age = current_time - timestamp
                alpha = max(0, 1 - age/MOVEMENT_DISAPPEAR_TIME)
                if alpha > 0:
                    trail_locations.append(pos)
                    trail_alphas.append(alpha)
            
            if trail_locations:
                ax_bee.scatter(
                    [p[0] for p in trail_locations],
                    [p[1] for p in trail_locations],
                    c="yellow", 
                    s=20,
                    alpha=trail_alphas,
                    edgecolors='none'
                )
            global prev_save_time
            plt.savefig(f'{bee_dir}/trail_{prev_save_time}ms-{current_time}ms.png', dpi=150, bbox_inches='tight')
            plt.close(fig_bee)
    for bee in sim.bees:
        bee.update_location_history(current_frame_id * FRAME_TIME)
    if (frame + 1) % (NUM_FRAMES // trail_save_num) == 0:
        save_bee_trails(frame+1, current_time)
        global prev_save_time
        prev_save_time = current_time + FRAME_TIME
    
    # Update the bees and nectars (and their displays)
    sim.update_bees()
    sim.update_nectar()
    locations = np.array([b.location_normalized for b in sim.bees])
    alphas = [0.7 if b.has_food else 0.2 for b in sim.bees]
    bee_scatter.set_offsets(locations)
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
    if (not frame_0_outputed): print (f"Render time: {render_duration:.2f}ms")

    total_duration = (time.perf_counter() - frame_start) * 1000
    if (not frame_0_outputed): print (f"Frame {frame} total time: {total_duration:.2f}ms\n")
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
         progress_callback=lambda i, n: print (f'\nProgress: {i+1}/{n} frames', end='\r'))
print ("\nAnimation saved! ")
plt.close('all')
os._exit(0)  # 强制终止进程
