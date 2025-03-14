'''
Living IoT: A Flying Wireless Platform on Live Insects
Bee flight simulation with a focus on the localization with multiple APs
'''
import numpy as np
import os, datetime, math, shutil, random, threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

if os.path.exists('sim.gif'):
    os.remove('sim.gif')
if os.path.exists('sim_figures'):
    shutil.rmtree('sim_figures')
if os.path.exists('sim_stats'):
    shutil.rmtree('sim_stats')


# Environment and Display Parameters
NUM_BEES = 10
NUM_FOOD_SOURCES = 5                        # randomly distributed in the field
HIVE_POS_NORM = (0.5, 0.1)                  # normalized
AP_POS_NORM = [(0.4, 0.0), (0.0, 0.5)]      # normalized
AP_NUM = len(AP_POS_NORM)
FIG_SIZE = (8, 8)
NECTAR_DISPLAY_SIZE = 50
NECTAR_EXPECTED_STORAGE = 15
NECTAR_EXPECTED_RECOVERY_TIME = 20

# AP Parameters
AP_PREAMBLES = [(1,0,1,0,1,0,1,0), (1,1,0,0,1,1,0,0)]
AP_SWEEP_T = 50
AP_PREAMBLE_T = 5   # time (ms) of preamble in a sweep <- self chosen, not from the paper
AP_PHASESHIFT_NUM = 50
assert (NUM_BEES * AP_NUM * AP_PHASESHIFT_NUM <= 10000)
AP_NUM_ANTENNAS = 2
AP_ANTENNA_LAYOUT_DIRECTION = [0, 3 * np.pi / 2]        # radians, counterclockwise from the positive x-axis
AP_FREQ = 915e6                                         # 915 MHz
AP_SIGNAL_STRENGTH = 28                                 # dBm
AP_SIGNAL_AMPLITUDE = 10**(AP_SIGNAL_STRENGTH / 20)     # Convert dBm to linear amplitude

# Simulation Parameters
FRAME_TIME = 50             # ms
SIM_TIME = 120000           # ms
NUM_FRAMES = round(SIM_TIME / FRAME_TIME)

# Bee Parameters
# bumblebee movement speed: x m/s (normally)
# which is (x / FIELD_SIZE) * normalized_edge_length m/s
# 1s = 20 frames, so 1 frame = (x / FIELD_SIZE / 20) * normalized_edge_length m
# When field size is 100, istance between two APs is 50, close to the paper configuration
FIELD_SIZE = 100
FIELD_ASPECT_RATIO = 1.0            # width / height
BUMBLEBEE_SPEED = 10                # m/s
STEP_SIZE = BUMBLEBEE_SPEED / FIELD_SIZE / 20
MIN_STEP = 0.75 * STEP_SIZE 
MAX_STEP = 1.25 * STEP_SIZE
STEP_NOISE = 0.25 * STEP_SIZE
NECTAR_DISCOVERY_DISTANCE = 5 * STEP_SIZE
NECTAR_EMPTY_DISCOVERY_DISTANCE = 10 * STEP_SIZE
OVERLAP_DISTANCE_ERROR = 3 * STEP_SIZE

# Debug print lock
debug_print_lock = threading.Lock()

# Functions
def to_meters(norm_pos):
    return [norm_pos[0] * FIELD_SIZE, norm_pos[1] * FIELD_SIZE / FIELD_ASPECT_RATIO]

def beamforming_phases_to_best_theta(phase_diffs):
    """
    Calculate the best θ for the given phases
    phase_diffs: in radians, unit: π
    """
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
    return max_amp, best_theta

def best_theta_to_target_angle(optimal_theta):
    """
    Calculate theoretical angle using paper's formula
    Discrete sampling errors and noises will cause deviation from the theoretical value
    optimal_theta: in radians, unit: π
    """
    if (optimal_theta > 1): 
        target_phase_diff = 2 - optimal_theta
    else: 
        target_phase_diff = -1 * optimal_theta
    target_phase_diff_normalized = target_phase_diff
    theta_cal = np.degrees(np.arccos(-1 * target_phase_diff_normalized))
    return theta_cal

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
        self.known_empty_sources = set()        # Local memory of empty food sources
        self.history_location = []              # Record location and time. Format: [(location, timestamp), ...]
        self.history_location_norm_est = []     # Record estimated location and time. Format: [(location, timestamp), ...]
        self.ap_angles_update_exp_smooth = 0.8
        self.signal_memory = []                 # Record the signal memory. Format: [(timestamp, amplitude), ...]
        self.signal_memory_preambles = []       # Record preamble signal's timestamp. Format: [((start_timestamp, ap_id), sampled_timestamp_amps), ...]
        self.signal_memory_ap_angles = []       # Record the AP angle from max amplitude timestamps. Format: [((max_amp_timestamp, ap_id), angle), ...]
        self.signal_memory_debug_count = 0
        self.signal_receive_plotted = False

        # Preamble parameters pre-calculation
        self.bit_duration = AP_PREAMBLE_T / len(AP_PREAMBLES[0]) # duration of a bit
        self.detected_bits = []
        self.detection_start_time = None

        # History statistics
        self.history_ap_angles_est = {}  # key: ap_id, value: [(angle, timestamp), ...]

        # Quick-access variables
        self.sweep_interval = AP_SWEEP_T * AP_NUM

    def reset_empty_food_knowledge(self): 
        '''
        Reset the knowledge of empty food sources
        Only for simulating the bee's behaviour, and not relevant to the paper
        '''
        self.known_empty_sources.clear()

    def update_location_history(self, frame_time): 
        '''
        Update the location history of the bees, useful for summarizing the bee's movement
        Only for simulating the bee's behaviour, and not relevant to the paper
        The length of history information is controlled by MOVEMENT_DISAPPEAR_TIME (used for plotting)
        Note that the history management is different from the signal memory management (AP_NUM + 1 sweeps preserved)
        '''
        self.history_location.append((self.location_normalized.copy(), frame_time))
        while (self.history_location[0][1] < frame_time - MOVEMENT_DISAPPEAR_TIME):
            self.history_location.pop(0)
        if (len(self.history_location_norm_est) > 0): 
            while (self.history_location_norm_est[0][1] < frame_time - MOVEMENT_DISAPPEAR_TIME):
                self.history_location_norm_est.pop(0)
        if (len(self.history_ap_angles_est.keys()) > 0): 
            for ap_id in self.history_ap_angles_est.keys(): 
                if (len(self.history_ap_angles_est[ap_id]) > 0): 
                    while (self.history_ap_angles_est[ap_id][0][1] < frame_time - MOVEMENT_DISAPPEAR_TIME):
                        self.history_ap_angles_est[ap_id].pop(0)
    
    def detect_preamble_signal(self, samples):
        '''
        Lightweight preamble detection to find the preamble signal and start signal processing
        Samples: [(timestamp, amplitude), ...]
        '''
        bit_duration = AP_PREAMBLE_T / len(AP_PREAMBLES[0])
        min_stable_time = 0.3 * bit_duration  # at least stable 0.3ms is valid
        
        # Step 1: Detect possible starting bit
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
                        ap_id = AP_PREAMBLES.index(detected_bits) # will throw an error if not found, which will be handled
                        
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
                        return None  # detected bits are not valid
                    
                elif abs(rel_amp - 1.0) >= 0.1: 
                    stable_start = None
                    current_max = None

        return None

    def find_angle_from_max_amplitude(self, signal_segment, ap_id): 
        '''
        Find the angle by detecting the max amplitude in a continuous signal segment
        According the paper, locate the max amplitude index, use linear interpolation to find phase difference
        signal_segment: [(timestamp, amplitude), ...]
        ap_id: the id of the AP just used to form return easier
        '''
        max_idx = np.argmax([s[1] for s in signal_segment])
        max_time = signal_segment[max_idx][0]
        percentage = (max_time - signal_segment[0][0]) / (AP_SWEEP_T - AP_PREAMBLE_T)
        best_theta = percentage * 2
        angle = best_theta_to_target_angle(best_theta) 
        return (max_time, ap_id), angle, signal_segment[max_idx][1]

    def plot_received_signal(self, bee_id, signal_data, mark=False):
        '''
        Plot the received signal from all APs
        '''
        if not signal_data: return
        
        # Plot the raw signal
        plt.figure(figsize=(12, 6))
        times = [s[0] for s in signal_data]
        amps = [s[1] for s in signal_data]
        plt.plot(times, amps, 'b-', alpha=0.3, label='Raw Signal')

        # Mark the preamble region and the sampling points
        if (mark): 
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
            rblues = ['#003333', '#006666', '#009999', '#00CCCC']
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
        Called by APs, to pass the simulated signal data to the bees
        In other words, APs will calculate the signal and pass the data to the bees
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
                if (self.signal_memory_debug_count < AP_NUM + 3): 
                    print (f"CROP the memory using prior knowledge (start from data id = {len(new_data)})")
            else:
                left, right = len(new_data), len(self.signal_memory)
                while left < right:
                    mid = (left + right) // 2
                    if (first_preserved_timestamp is not None) and (self.signal_memory[mid][0] < first_preserved_timestamp):
                        left = mid + 1
                    else:
                        right = mid
                self.signal_memory = self.signal_memory[left:] if left < len(self.signal_memory) else []
                if (self.signal_memory_debug_count < AP_NUM + 3): 
                    print (f"CROP the memory using binary search (start from {left})")

        # Cut the preambles and angles memory at the same time
        # Not too costly, as each sweep period => only 1 statistics saved, 
        # so just perform linear search
        self.signal_memory_preambles = [p for p in self.signal_memory_preambles if p[0][0] >= first_preserved_timestamp]
        self.signal_memory_ap_angles = [a for a in self.signal_memory_ap_angles if a[0][0] >= first_preserved_timestamp]

        # Search for new preambles in the new data
        # "locate preamble2 in S_(i+T) .. S_(3T) at i" indicates that preamble pairs should have at least T time gap between them
        # According to the paper, the new preamble detection can start from the last preamble timestamp even recorded
        preamble_search_index = len(self.signal_memory) - len(new_data)
        preamble_detection = None
        angle_detection = None
        while (self.signal_memory_preambles and preamble_search_index < len(self.signal_memory) and
               self.signal_memory[preamble_search_index][0] < self.signal_memory_preambles[-1][0][0] + AP_SWEEP_T): 
            preamble_search_index += 1
        
        # The preamble must be found before the phase shift detection
        if (preamble_search_index < len(self.signal_memory)):
            result = self.detect_preamble_signal(self.signal_memory[preamble_search_index:])
            if (result):
                preamble_info, sampled_timestamp_amps, end_idx = result
                preamble_detection = preamble_info[1]
                self.signal_memory_preambles.append((preamble_info, sampled_timestamp_amps))
                
                # Find the first timestamp that is T time after the previous preamble, 
                # and all the signals in between are saved for phase shift detection
                signal_start_idx = preamble_search_index + end_idx
                sweep_end_time = preamble_info[0] + AP_SWEEP_T
                signal_segment = []
                for i in range(signal_start_idx, len(self.signal_memory)):
                    ts, amp = self.signal_memory[i]
                    if ts > sweep_end_time: break
                    signal_segment.append( (ts, amp) )
                
                # Calculate the angle from the max amplitude
                max_amp_sample, angle, point_amp = self.find_angle_from_max_amplitude(signal_segment, preamble_detection)
                self.signal_memory_ap_angles.append((max_amp_sample, angle, point_amp))

                # Save the phase shift angle estimation to history records
                if (preamble_detection not in self.history_ap_angles_est):
                    self.history_ap_angles_est[preamble_detection] = [(angle, preamble_info[0])]
                else: 
                    last_angle = self.history_ap_angles_est[preamble_detection][-1][0]
                    self.history_ap_angles_est[preamble_detection].append((self.ap_angles_update_exp_smooth * last_angle + (1 - self.ap_angles_update_exp_smooth) * angle, preamble_info[0]))
                angle_detection = self.history_ap_angles_est[preamble_detection][-1][0]

                # Use the latest angle estimation records
                location_est = self.get_estimated_location()
                if (location_est):
                    self.history_location_norm_est.append((location_est, preamble_info[0]))

        # Signal receive debug logging (only applicable to the first few sweeps)
        if (self.signal_memory_debug_count > AP_NUM+1 and 
            self.signal_memory_debug_count <= 2*AP_NUM+1 and 
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
                if (angle_detection):
                    print (f"│  └─ Angle detected: {round(angle_detection, 2)}°")
            print (f"└─ ap_id: (max_amp_ts, angle): \n  │ {[f'{ts_and_ap_id[1]}: {round(ts_and_ap_id[0], 2)}ms, {round(angle, 2)}°' for ts_and_ap_id, angle, _ in self.signal_memory_ap_angles]}")
            if (len(self.history_ap_angles_est.keys()) > 0): 
                character = '└─'
                if (len(self.history_location_norm_est) > 0):
                    character = '├─'
                print (f"  {character} AP angle est.: \n  │ {[f'{ap_id}: {round(angle_and_ts[-1][0], 2)}°, {round(angle_and_ts[-1][1], 2)}ms' for ap_id, angle_and_ts in self.history_ap_angles_est.items()]}")
                if (len(self.history_location_norm_est) > 0):
                    print (f"  └─ AP location norm. est.: \n    {[f'{ts}ms: ({round(location[0], 2)}, {round(location[1], 2)})' for location, ts in self.history_location_norm_est]}")
            self.plot_received_signal(self.bee_id, self.signal_memory, mark = True)
            self.signal_receive_plotted = True

    def get_estimated_location(self):
        '''
        Estimate the location of the current bee using the latest angle estimation records
        '''
        # At least 2 APs are needed to estimate the location
        valid_aps = [ap_id for ap_id in self.history_ap_angles_est if ap_id < len(APs)]
        if len(valid_aps) < 2: return None
        location_estimated = self.signal_memory_debug_count > AP_NUM+1 and self.signal_memory_debug_count <= 2*AP_NUM+1

        # Only consider the direction with positive angle
        # Future work: When there are multiple solutions, especially when num of antennas is more than 2, 
        # improved algorithms should be designed to select the best one
        ap_data = []
        for ap_id in valid_aps:
            ap = AP_POS_NORM[ap_id]
            ant_dir = AP_ANTENNA_LAYOUT_DIRECTION[ap_id]
            measured_angle = np.radians(self.history_ap_angles_est[ap_id][-1][0].item())
            angle = ant_dir + measured_angle
            ap_data.append((ap, angle))

        if (location_estimated): 
            print  (f"Bee {self.bee_id} estimate location: ")
            for ap_loc, angle in ap_data:
                print  (f"├─ AP location: {[round(ap_loc[0], 3), round(ap_loc[1], 3)]}, angle: {round(angle, 3)}π")

        # Calculate all possible intersections between each pair of APs
        intersections = []
        for i in range(len(ap_data)):
            for j in range(i+1, len(ap_data)):
                (p1, angle1), (p2, angle2) = ap_data[i], ap_data[j]
                intersect = self.calculate_intersection(p1, angle1, p2, angle2)
                if intersect and (0 <= intersect[0] <= FIELD_SIZE) and (0 <= intersect[1] <= FIELD_SIZE / FIELD_ASPECT_RATIO): 
                    if (location_estimated):
                        print (f"├─ AP {ap_data[i][0]} and AP {ap_data[j][0]} intersect at {[round(intersect[0].item(), 3), round(intersect[1].item(), 3)]}")
                    intersections.append(intersect)
        
        if not intersections: 
            if (location_estimated):
                print (f"└─ No valid intersection found")
            return None
        
        # Algorithm: Average all the intersections between AP pairs
        # Normalize the average intersection point
        avg_point = np.mean(intersections, axis=0)
        point_norm = (avg_point[0]/FIELD_SIZE, avg_point[1]/(FIELD_SIZE/FIELD_ASPECT_RATIO))
        if (location_estimated):
            print (f"└─ Bee {self.bee_id} estimated location norm.: {[round(i.item(), 3) for i in point_norm]}")
        return point_norm

    def calculate_intersection(self, p1, theta1, p2, theta2): 
        '''
        Calculate the intersection point of two lines given two points and two rays from them
        p1, p2: the coordinates of the two points (normalized)
        theta1, theta2: the angles of the two rays (in radians)
        '''
        p1_phy = (p1[0] * FIELD_SIZE, p1[1] * (FIELD_SIZE / FIELD_ASPECT_RATIO))
        p2_phy = (p2[0] * FIELD_SIZE, p2[1] * (FIELD_SIZE / FIELD_ASPECT_RATIO))
        
        # Build the ray equations and solve it
        A = np.array([
            [np.cos(theta1), -np.cos(theta2)],  
            [np.sin(theta1), -np.sin(theta2)]  
        ])
        b = np.array([p2_phy[0] - p1_phy[0], p2_phy[1] - p1_phy[1]])
        try:
            t = np.linalg.solve(A, b)
            t1, t2 = t[0], t[1]
            if t1 >= 0 and t2 >= 0:
                intersect_x = p1_phy[0] + t1 * np.cos(theta1)
                intersect_y = p1_phy[1] + t1 * np.sin(theta1)
                norm_x = intersect_x / FIELD_SIZE
                norm_y = intersect_y / (FIELD_SIZE / FIELD_ASPECT_RATIO)
                if 0.01 <= norm_x <= 0.99 and 0.01 <= norm_y <= 0.99:
                    return (intersect_x, intersect_y)
        except np.linalg.LinAlgError: pass
        return None



class AP: 
    '''
    Access Point class. Access points will be in charge of sending signals to the bees
    and sampling the signals with pre-defined settings, illustrating the key ideas in the paper
    '''
    def __init__(self, ap_id, location, antenna_num, frequency, preamble_signal, antenna_layout_direction = 0):
        '''
        Initialize the AP
        ap_id: the id of the AP
        location: the location of the AP (in meters)
        antenna_num: the number of antennas
        frequency: the frequency of the signal (in Hz)
        preamble_signal: the preamble signal bit pattern
        antenna_layout_direction: the direction of the antennas (in radians)
        '''
        self.ap_id = ap_id
        self.location = np.array(location)
        self.location_normalized = self.location / FIELD_SIZE
        self.preamble = preamble_signal
        self.signal_start = 0  # in ms
        self.frequency = frequency  # in Hz
        self.antenna_num = antenna_num
        self.antenna_layout_direction = antenna_layout_direction
        self.wavelength = 3e8 / frequency
        
        # CRITICAL: According to the paper, say there are 2 antennas, 
        # When the first antenna has a longer distance to the target, namely Φ, 
        # There will be a relationship: arccos(Φ / d) = θ, and θ is within [0, π/2] (INSTEAD OF arcsin)
        # Half wavelength spacing between antennas, e.g., should be around 33cm for 915MHz in paper
        self.antenna_spacing = self.wavelength / 2  
        antenna_layout_direction_vector = np.array([np.cos(antenna_layout_direction), np.sin(antenna_layout_direction)])
        self.antenna_locations = [
            self.location + self.antenna_spacing * (i - (self.antenna_num-1)/2) * antenna_layout_direction_vector
            for i in range(self.antenna_num)
        ]

        # NLOS attenuation map that simulates the signal attenuation due to ONLY the obstacles from other APs, the hive, and the food sources
        # Future work: More sophisticated attenuation mechanism should be designed
        self.attenuation_map = None


    def load_attenuation_map(self):
        '''
        Load attenuation map when initializing the AP. 
        Each discretized cell in the map has a corresponding attenuation value
        '''
        attenuation_map = {}
        nlos_attenuation_path = os.path.join(os.path.dirname(__file__),"sim_stats", f"AP_{self.ap_id}", 'nlos_attenuation.csv')
        if (not os.path.exists(nlos_attenuation_path)):
            raise FileNotFoundError(f"NLOS attenuation map file not found: {nlos_attenuation_path}")
        with open(nlos_attenuation_path, 'r') as f:
            for line in f:
                x_str, y_str, atten = line.strip().split(',')
                x = int(float(x_str))
                y = int(float(y_str))
                attenuation_map[(x, y)] = float(atten)
        self.attenuation_map = attenuation_map
    
    def sample_signal(self, timestamp, location=None): 
        '''
        Sample the signal at a specific timestamp. 
        According to the paper, each AP sweeps in turns, so every AP sweeps once in AP_NUM * AP_SWEEP_T ms
        '''
        
        # Calculate the phase difference for each antenna: 
        # Firstly, find the process that corresponds to the timestamp t in the sweep cycle
        phase_diffs = self.calculate_beamforming_phase(location) if (location is not None and len(location) > 0) else [0]*(self.antenna_num-1)
        sweep_start = self.signal_start + ((timestamp - self.signal_start) // (AP_NUM * AP_SWEEP_T)) * (AP_NUM * AP_SWEEP_T)
        if timestamp < sweep_start + AP_PREAMBLE_T:  
            # Within preamble stage: Sample a signal according to the preamble bit pattern
            return AP_SIGNAL_AMPLITUDE * AP_PREAMBLES[self.ap_id][
                int((timestamp - sweep_start) / AP_PREAMBLE_T * len(AP_PREAMBLES[self.ap_id]))
            ]
        else:  
            # Within beamforming scanning stage: Signals from antennas are superposed in the complex domain
            # Amplitude is the normalized result of the complex superposition
            phase_index = AP_PHASESHIFT_NUM * (timestamp - sweep_start - AP_PREAMBLE_T) / (AP_SWEEP_T - AP_PREAMBLE_T)
            ap_theta = 2 * np.pi * phase_index / AP_PHASESHIFT_NUM
            complex_sum = 1.0
            for i in range(self.antenna_num - 1):
                phase = ap_theta + phase_diffs[i]
                complex_sum += np.exp(1j * phase)
            amplitude = np.abs(complex_sum)
            return amplitude
    
    def generate_signal(self, sweep_start, location=None):
        '''
        Generate a signal given the sweep start time for a sweep period for a specific location
        '''
        preamble_samples = 10 * len(AP_PREAMBLES[0])
        scan_samples = 2 * AP_PHASESHIFT_NUM
        preamble_ts = np.linspace(sweep_start, sweep_start+AP_PREAMBLE_T, preamble_samples, endpoint=False)
        scan_ts = np.linspace(sweep_start+AP_PREAMBLE_T, sweep_start+AP_SWEEP_T, scan_samples, endpoint=False)
        timestamps = np.concatenate([preamble_ts, scan_ts]).tolist()
        samples = []
        for t in timestamps: 
            samples.append(self.sample_signal(t, location))
        return timestamps, samples
    
    def transmit_signal(self, bees, timestamp):
        '''
        Transmit the signal to a list of bees (i.e., simulate the signal receiving by the bees) since a timestamp
        The transmitted signal lasts for AP_SWEEP_T ms
        '''
        # Check if the current timestamp is within the transmission window
        if (self.signal_start is None or 
            timestamp < self.signal_start or 
            (timestamp - self.signal_start) % (AP_NUM * AP_SWEEP_T) >= AP_SWEEP_T):
            return 0
        
        # Generate the time axis for the current sweep period, and send the signal to all bees
        sweep_start = self.signal_start + ((timestamp - self.signal_start) // AP_SWEEP_T) * AP_SWEEP_T
        for bee in bees: 
            timestamps, raw_amps = self.generate_signal(sweep_start, bee.location_normalized)
            attenuation = self.calculate_attenuation(bee.location_normalized)
            amp_scale = 10**(-attenuation/20)
            samples = [s * amp_scale for s in raw_amps]
            bee.receive_signal(timestamps, samples)

    def test_signal_characteristics(self):
        '''
        Generate preamble and actual signal analysis with separated time scales
        The AP time-dependent phase shift and amplitude change is considered
        '''

        # Find the signal period
        period_ns = 1e9 / self.frequency        # in ns
        preamble_duration_ms = AP_PREAMBLE_T    # in ms
        num_periods = 3
        signal_duration_ns = (num_periods+1) * period_ns  # in ns, one more period allow phase shift part finish drawing
        
        # Dynamic sampling settings
        samples_preamble = 100  # total preamble samples in the preamble stage
        sampling_rate = 20      # sampling rate in the sweeping stage
        
        # Generate test signal data
        preamble_t = np.linspace(self.signal_start, self.signal_start + preamble_duration_ms, samples_preamble, endpoint=False)
        sweep_t = np.linspace(self.signal_start + preamble_duration_ms, self.signal_start + AP_SWEEP_T, sampling_rate * AP_PHASESHIFT_NUM, endpoint=False)
        preamble_data = [self.sample_signal(t) for t in preamble_t]
        phase_shift = [self.sample_signal(t) for t in sweep_t]

        # Create dual-view layout for preamble and sweeping signal without a position provided
        _ = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        
        # Preamble signal view (stepped line chart)
        ax1 = plt.subplot(gs[0])
        preamble_level_1 = np.max(preamble_data).item()
        preamble_digital = np.array(preamble_data) / preamble_level_1
        ax1.step(preamble_t, preamble_digital, where='post', color='#FF4500', linewidth=1.5)
        ax1.set_title(f"Preamble Digital Signal ({preamble_duration_ms}ms)\n"
                      f"(Record start t: {self.signal_start}ms)")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Digital Level")
        ax1.grid(True, axis='y', linestyle=':')
        ax1.set_xlim(self.signal_start, self.signal_start + preamble_duration_ms)
        ax1.set_ylim(-0.1, 1.1)
        
        # Sweeping signal view (continuous line chart)
        ax2 = plt.subplot(gs[1])
        relative_t = sweep_t - self.signal_start - preamble_duration_ms
        ax2.plot(relative_t, phase_shift, 'b-', alpha=0.8)

        # Create the overall chart
        ax2.set_title(f"Phase Shift ({self.frequency/1e6}MHz) (AP start t: {self.signal_start}ms)\n"
                      f"(Record start t: {self.signal_start}ms + {preamble_duration_ms}ms (preamble) @ 0ms in the chart)")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, which='both', alpha=0.4)
        ax2.set_xlim(0, AP_SWEEP_T - preamble_duration_ms)
        ax2.set_ylim(-0.1 * AP_NUM_ANTENNAS, 1.1 * AP_NUM_ANTENNAS)
        output_dir = f'sim_figures/AP_{self.ap_id}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "signal_characteristics.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Print the statistics debug logs
        print (f"\nAP{self.ap_id} Statistics:")
        print (f"├─ Location: {[round(i.item(), 3) for i in self.location]}")
        print (f"├─ Preamble Duration: {preamble_duration_ms}ms")
        print (f"├─ Signal (sweep) Duration: {signal_duration_ns:.2f}ns")
        print (f"├─ Phaseshifts per sweep: {AP_PHASESHIFT_NUM}")
        print (f"├─ Carrier Frequency: {self.frequency/1e6}MHz")
        print (f"├─ Single Period: {period_ns:.4f}ns")
        print (f"├─ Wavelength: {self.wavelength:.4f}m")
        print (f"├─ Antenna Spacing: {self.antenna_spacing:.4f}m (half wavelength)")
        print (f"├─ Antenna Layout Direction: {round(self.antenna_layout_direction, 3)}π")
        for i in range(self.antenna_num):
            print (f"│   {'├─' if i < self.antenna_num - 1 else '└─'} Antenna {i} Location: {[round(i.item(), 3) for i in self.antenna_locations[i]]}")
        print (f"└─ 3 Cycles Duration: {signal_duration_ns:.2f}ns")

    def test_signal_charactertistics_at_location(self, norm_pos):
        '''
        Verify the signal phase shift characteristics at a specific location
        norm_pos: the normalized position of the location (in meters) to sample the signal
        '''
        location = np.array(norm_pos)
        timestamps = np.arange(self.signal_start, self.signal_start + AP_SWEEP_T, 0.1)  # 0.1ms resolution
        original = []
        attenuated = []
        phases = []

        for t in timestamps:

            # Original signal (no attenuation)
            amp = self.sample_signal(t, location)
            original.append(amp)
            
            # Actual received signal (with beamforming and attenuation)
            # Simulate the phase shift of the antennas according to the paper
            equivalent_signal_start = self.signal_start + AP_SWEEP_T * ((t - self.signal_start) // AP_SWEEP_T)
            attenuation = self.calculate_attenuation(location)
            amp_scale = 10 ** (-(AP_SIGNAL_AMPLITUDE - attenuation)/20)
            actual_amp = self.sample_signal(t, location) * amp_scale
            attenuated.append(actual_amp)
            phase_shift = -np.pi/2 + (np.pi / AP_PHASESHIFT_NUM) * int(AP_PHASESHIFT_NUM * (t - equivalent_signal_start - AP_PREAMBLE_T) / (AP_SWEEP_T - AP_PREAMBLE_T))
            phases.append(phase_shift)

        # Figure 1: Original signal figure
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, original, label='Original Amplitude')
        plt.title(f"AP{self.ap_id} Signal Components @ ({norm_pos[0]:.2f}, {norm_pos[1]:.2f})")
        plt.ylabel("Amplitude (V)")
        plt.legend()
        
        # Figure 2: Attenuated signal figure
        # Currently, this figure looks similar to the original signal figure, as the attenuation is constant for a given location
        # In the future, we will add random noises and other time-varying factors to the attenuation
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, attenuated, color='orange', label='Attenuated Amplitude')
        plt.ylabel("Amplitude (V)")
        plt.legend()
        
        # For each timestamp, calculate the theta value
        # Case 1: Preamble phase -> fixed theta value (in the end, we will not display the preamble stage's phase shift)
        # Case 2: Sweep phase -> continuous change in the scan phase
        theta_values = []
        for t in timestamps:
            if t < self.signal_start + AP_PREAMBLE_T: 
                theta_values.append(-np.pi/2)
            else:
                scan_progress = (t - self.signal_start - AP_PREAMBLE_T) / (AP_SWEEP_T - AP_PREAMBLE_T)
                theta = -np.pi/2 + scan_progress * np.pi
                theta_values.append(theta)

        # Figure 3: Phase change figure
        plt.subplot(3, 1, 3)
        for j in range(2, self.antenna_num+1):
            phases = [(j-1) * np.pi * np.sin(theta) for theta in theta_values]  # 以π为单位
            sweep_mask = [t >= self.signal_start + AP_PREAMBLE_T for t in timestamps]
            plt.plot(np.array(timestamps)[sweep_mask], 
                    np.array(phases)[sweep_mask], 
                    linewidth=1.5,
                    label=f'Antenna {j} Phase')

        # Set y-axis ticks as multiples of π
        max_phase = (self.antenna_num-1)
        y_ticks = np.arange(-max_phase, max_phase+1)
        plt.yticks(y_ticks, [f'{x}' if x !=0 else '0' for x in y_ticks])
        plt.xlabel("Time (ms)")
        plt.ylabel("Phase (-π to π)")
        plt.grid(alpha=0.3)

        # Add preamble region annotation
        plt.axvspan(self.signal_start, self.signal_start + AP_PREAMBLE_T, color='gray', alpha=0.2, label='Preamble stage')
        plt.legend()
        plt.tight_layout()
        output_dir = f'sim_figures/AP_{self.ap_id}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"signal_charactertistics_at_location_{norm_pos[0]:.2f}_{norm_pos[1]:.2f}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def test_beamforming_best_theta(self):
        """
        Generate the beamforming maximum amplitude mapping to phase shifts
        """
        # Create the test grid for recording the maximum amplitude and the best phase for each discretized location
        grid_size = 201 # handle the edge case
        x = np.linspace(0, FIELD_SIZE, grid_size, endpoint=True)
        y = np.linspace(0, FIELD_SIZE/FIELD_ASPECT_RATIO, grid_size, endpoint=True)
        max_amplitudes = np.zeros((grid_size, grid_size))
        best_phases = np.zeros((grid_size, grid_size))
        phase_diffs = np.zeros((grid_size, grid_size, self.antenna_num-1))
        for i in range(grid_size):
            for j in range(grid_size):
                norm_pos = np.array([x[i]/FIELD_SIZE, y[j]/FIELD_SIZE])
                phase_diffs[i,j] = self.calculate_beamforming_phase(norm_pos)

        # Signal superposition calculation: Iterate the theta range (0 to 2π)
        theta_values = np.linspace(0, 2*np.pi, AP_PHASESHIFT_NUM, endpoint=False)
        for theta in theta_values: 
            complex_sum = np.ones((grid_size, grid_size), dtype=complex)
            for k in range(self.antenna_num - 1):
                phase = theta + phase_diffs[:,:,k]
                complex_sum += np.exp(1j * phase)
            amplitudes = np.abs(complex_sum)

            # Update the maximum amplitude if possible
            mask = amplitudes > max_amplitudes
            max_amplitudes[mask] = amplitudes[mask]
            best_phases[mask] = theta % (2*np.pi)

        # Dynamic test point generation
        max_attempts = 10
        test_points = []
        for _ in range(max_attempts): 
            # continue  ## uncomment this line to test the bee hive location
            theta = np.random.uniform(0, 2*np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)])
            
            # Calculate AP center location (normalized)
            ap_center = self.location / FIELD_SIZE
            max_step = min(
                (1 - ap_center[0])/direction[0] if direction[0]>0 else ap_center[0]/-direction[0],
                (1 - ap_center[1])/direction[1] if direction[1]>0 else ap_center[1]/-direction[1],
                key=abs
            ) if not np.allclose(direction, 0) else 0
            if max_step < 0.2: continue
            
            # Generate two test points (distance ≥ 0.2 normalized)
            step1 = np.random.uniform(0.1, max_step-0.1)
            step2 = step1 + 0.2 + np.random.uniform(0, 0.1)
            if step2 > max_step: continue
            p1 = ap_center + direction * step1
            p2 = ap_center + direction * step2
            test_points = [p1, p2]
            break
        else: # fallback to the default test points
            test_points = [np.array(HIVE_POS_NORM) ]
            # test_points = [
            #     np.array([self.location_normalized[0] + 0.1, self.location_normalized[1]]),
            #     np.array([self.location_normalized[0] + 0.2, self.location_normalized[1]])
            # ]

        # Store the test point information
        test_info = []
        for point_idx, norm_pos in enumerate(test_points):
            print(f"\nTestpoint {point_idx+1} optimal phase calculation:")
            print(f"├─ Normalized coordinates: ({norm_pos[0]:.3f}, {norm_pos[1]:.3f})")
            pos_m = norm_pos * FIELD_SIZE
            print(f"├─ Actual coordinates: ({pos_m[0]:.1f}m, {pos_m[1]:.1f}m)")
            print("├─ Antenna distance to test point:")
            for i, ant_loc in enumerate(self.antenna_locations):
                dist = np.linalg.norm(pos_m - ant_loc)
                print(f"│   {'→' if i==0 else ' '} Antenna {i}: {dist:.2f}m" + 
                     f"{' (referenced distance)' if i==0 else ''}")
            phase_diffs = self.calculate_beamforming_phase(norm_pos)
            max_amp, best_theta = beamforming_phases_to_best_theta(phase_diffs)
            print(f"├─ Antenna phase differences: {[round((pd/np.pi).item(), 3) for pd in phase_diffs]}π")
            atten = self.calculate_attenuation(norm_pos)
            print(f"└─ Optimal θ: {best_theta/np.pi:.3f}π")
            print(f"   ├─ Maximum amplitude: {max_amp:.3f}")
            print(f"   └─ Total attenuation: {atten:.2f}dB")
            test_info.append({
                'location': norm_pos,
                'phase_diffs': [round(float(pd/np.pi),3) for pd in phase_diffs], 
                'best_theta': round(float(best_theta/np.pi),3),
                'max_amp': round(float(max_amp),3), 
                'atten': round(float(atten),2)
            })

        # Create the figure for the optimal phase map
        plt.figure(figsize=(12, 10))
        blues = plt.cm.Blues(np.linspace(0.1, 0.9, 256 + 1))
        blues[:, 3] = np.linspace(0.1, 0.6, 256 + 1)
        cmap = ListedColormap(blues)

        # Calculate the plot range and configure the colorbar
        img = plt.imshow(best_phases.T, extent=[0, FIELD_SIZE, 0, FIELD_SIZE / FIELD_ASPECT_RATIO],
                  origin='lower', cmap=cmap, aspect='equal',
                  vmin=0, vmax=2*np.pi, interpolation='bilinear')
        cbar = plt.colorbar(img, 
                           label='Optimal Phase Shift θ (rad)',
                           ticks=np.linspace(0, 2*np.pi, 5, endpoint=True),
                           extend='both')
        cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        cbar.outline.set_edgecolor('black')

        # Add static environment markers, arrows and angle annotations to the test points
        self.draw_plot_environment_markers()
        ax = plt.gca()
        ap_x, ap_y = self.location
        ax.plot(ap_x, ap_y, 'r^', markersize=15, zorder=5)
        ant_dir = self.antenna_locations[1] - self.antenna_locations[0]
        ant_dir /= np.linalg.norm(ant_dir)
        ax.arrow(ap_x, ap_y, ant_dir[0] * 0.05 * FIELD_SIZE, ant_dir[1] * 0.05 * FIELD_SIZE, 
                head_width=2, head_length=3, fc='black', ec='black', zorder=5, 
                label='Antenna direction')
        ax.set_xlim(0, FIELD_SIZE)
        ax.set_ylim(0, FIELD_SIZE / FIELD_ASPECT_RATIO)
        for i, test_point in enumerate(test_info):
            px, py = test_point['location'] * FIELD_SIZE
            dx, dy = px - ap_x, py - ap_y
            dx, dy = dx / np.linalg.norm([dx, dy]) * 0.1 * FIELD_SIZE, dy / np.linalg.norm([dx, dy]) * 0.1 * FIELD_SIZE
            # ax.arrow(ap_x, ap_y, dx, dy, head_width=2, head_length=3, fc='black', ec='black', linestyle='--') ## The actual direction
            dx_rot = dx * np.cos(self.antenna_layout_direction) + dy * np.sin(self.antenna_layout_direction)
            dy_rot = -dx * np.sin(self.antenna_layout_direction) + dy * np.cos(self.antenna_layout_direction)
            theta_geo = np.degrees(np.arctan2(dy_rot, dx_rot))
            theta_cal = best_theta_to_target_angle(test_point['best_theta'])
            x = test_point['location'][0] * FIELD_SIZE
            y = test_point['location'][1] * FIELD_SIZE
            if i == 0: plt.scatter(x, y, s=120, c='white', edgecolors='red', linewidths=1.5, zorder=4, label=f'Test Points')
            else: plt.scatter(x, y, s=120, c='white', edgecolors='red', linewidths=1.5, zorder=4)
            va = 'bottom' if y < FIELD_SIZE*0.8 else 'top'
            plt.text(x, y+3, 
                     f"θ*={test_point['best_theta']}π, ΔΦ={test_point['phase_diffs']}π\nθ_geo={theta_geo:.1f}°, θ_cal={theta_cal:.1f}° OR {-1 * theta_cal:.1f}°", 
                    color='white', fontsize=9,
                    ha='center', va=va, 
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            
            # two possible theta_cal directions given the same theta_cal value
            for i in [(1, '#555555'), (-1, '#bbbbbb')]: 
                dx = FIELD_SIZE * 0.2 * np.cos(theta_cal * i[0] * np.pi / 180 + self.antenna_layout_direction)
                dy = FIELD_SIZE * 0.2 * np.sin(theta_cal * i[0] * np.pi / 180 + self.antenna_layout_direction)
                ax.arrow(ap_x, ap_y, dx, dy, head_width=2, head_length=3, fc=i[1], ec=i[1], linestyle='--', label=f'θ_cal = {i[0] * theta_cal:.1f}°')
        
        # Add the legends and save the figure
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels.append(l)
                unique_handles.append(h)
        ax.legend(handles=unique_handles, labels=unique_labels, loc='upper right', fontsize=8)
        plt.title(f"AP{self.ap_id} Optimal θ Map (antenna num: {self.antenna_num}, direction: {self.antenna_layout_direction}°)\n(Phase shifts per sweep: {AP_PHASESHIFT_NUM})")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        save_path = os.path.join(os.path.dirname(__file__), f"sim_figures/AP_{self.ap_id}/beamforming_best_theta.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def calculate_attenuation(self, location_normalized):
        '''
        Retrieve the attenuation from the attenuation map
        location_normalized: the normalized location (in meters) to sample the attenuation
        '''
        x = int(location_normalized[0] * FIELD_SIZE)
        y = int(location_normalized[1] * FIELD_SIZE / FIELD_ASPECT_RATIO)
        return self.attenuation_map.get((x, y), 0.0)

    def calculate_beamforming_phase(self, location_normalized): 
        '''
        Calculate the phase shifts for each antenna given the referenced antenna (the first one)
        location_normalized: the normalized location (in meters) to calculate the phase shift
        '''
        location = location_normalized * FIELD_SIZE
        distances = [np.linalg.norm(location - ant) for ant in self.antenna_locations]
        phase_diffs = [(d - distances[0]) * (2 * np.pi / self.wavelength) for d in distances[1:]]
        return np.array(phase_diffs)

    def draw_plot_environment_markers(self):
        '''
        Unified environment markers plotting function
        It will plot the current AP, other APs and the hive
        '''
        # Current and other APs
        ap_loc = to_meters(self.location_normalized)
        plt.scatter(ap_loc[0], ap_loc[1], marker='^', s=200, c='red', edgecolors='white', label=f'AP {self.ap_id}')
        for ap in APs:
            if ap.ap_id != self.ap_id:
                other_loc = to_meters(ap.location_normalized)
                plt.scatter(other_loc[0], other_loc[1], marker='^', s=120, c='red', alpha=0.5)
        
        # The hive and food sources
        hive_pos = to_meters(HIVE_POS_NORM)
        plt.scatter(hive_pos[0], hive_pos[1], marker='s', s=120, c='green', alpha=0.5)
        for fs in food_sources:
            fs_pos = to_meters(fs.location_normalized)
            plt.scatter(fs_pos[0], fs_pos[1], marker='o', s=80, c='blue', alpha=0.5)
        
        # Add the legends, do not close the plot and 
        # the caller will continue to draw the rest of the figure
        plt.legend(loc='upper right', framealpha=0.9)

    def get_phase_at_time(self, timestamp):
        '''
        Get the phase configuration at a specific timestamp
        timestamp: the timestamp to get the phase configuration (in ms)
        '''
        sweep_index = int((timestamp - self.signal_start) // AP_SWEEP_T)
        theta_values = np.linspace(0, 2*np.pi, AP_PHASESHIFT_NUM, endpoint=False)
        return theta_values[sweep_index % len(theta_values)]

class Simulation:
    def __init__(self, food_sources, num_bees = NUM_BEES): 
        '''
        Initialize the simulation
        food_sources: the list of food source locations
        num_bees: the number of bees
        '''
        self.start_time = datetime.datetime.now()
        self.food_sources = food_sources
        self.bees = [Bee(i, food_sources) for i in range(num_bees)]
        self.frame_count = 0

    def update_nectar(self):
        '''
        Update the nectar of the food sources when it runs out
        Just for the visualization purpose and does not relate to the paper
        '''
        self.frame_count += 1
        for _, fs in enumerate(self.food_sources):
            if fs.nectar == -1: 
                if fs.recovery_timer > 0:
                    fs.recovery_timer -= 1
                else:
                    fs.nectar = round((0.8 + 0.4 * random.random()) * NECTAR_EXPECTED_STORAGE)
                    fs.recovery_timer = 0

    def update_bees(self):
        '''
        Update the bees when the simulation runs
        Just for the visualization purpose and does not relate to the paper
        '''
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
                
                # Find a new target food if the current target food is not available
                if not bee.target_food:
                    potential_sources = [
                        fs for fs in self.food_sources 
                        if fs not in bee.known_empty_sources
                    ]
                    if potential_sources:
                        distances = [np.linalg.norm(fs.location_normalized - bee.location_normalized) for fs in potential_sources]
                        bee.target_food = potential_sources[np.argmin(distances)]
                    else:
                        bee.reset_empty_food_knowledge()
                
                # Move towards the target food if it exists
                if bee.target_food:
                    direction = bee.target_food.location_normalized - bee.location_normalized
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        step_size = np.clip(STEP_SIZE * distance, MIN_STEP, MAX_STEP)
                        step = (direction / distance) * step_size
                        bee.location_normalized += step + np.random.normal(0, STEP_NOISE, 2)

                    # Arrived at the target food and collect the nectar
                    if np.linalg.norm(bee.location_normalized - bee.target_food.location_normalized) < OVERLAP_DISTANCE_ERROR:
                        if bee.target_food.nectar > 0:
                            bee.target_food.nectar -= 1
                            bee.has_food = True
                        else:
                            bee.known_empty_sources.add(bee.target_food)
                        bee.target_food = None
            else:
                # Move back to hive if the bee has food
                direction = np.array(HIVE_POS_NORM) - bee.location_normalized
                distance = np.linalg.norm(direction)
                if distance > 0:
                    step_size = np.clip(STEP_SIZE * distance, MIN_STEP, MAX_STEP)
                    step = (direction / distance) * step_size
                    bee.location_normalized += step + np.random.normal(0, STEP_NOISE, 2)
                
                # Arrived at hive and drop the food
                if np.linalg.norm(bee.location_normalized - HIVE_POS_NORM) < OVERLAP_DISTANCE_ERROR:
                    bee.has_food = False

        for _, fs in enumerate(self.food_sources): 
            # If the food source runs out, set it to be empty to recover after some time
            if fs.nectar == 0: 
                fs.nectar = -1
                fs.recovery_timer = 1000 / FRAME_TIME * NECTAR_EXPECTED_RECOVERY_TIME * (0.8 + 0.4 * random.random())



# Generate APs for the simulation
# All APs are coarsely synchronized using TDMA -> set their signal_start to reflect this
APs = []
for i, pos in enumerate(AP_POS_NORM):
    new_AP = AP(i, (pos[0] * FIELD_SIZE, pos[1] * FIELD_SIZE / FIELD_ASPECT_RATIO),
                  AP_NUM_ANTENNAS, AP_FREQ, AP_PREAMBLES[i], AP_ANTENNA_LAYOUT_DIRECTION[i])
    new_AP.signal_start = -i * AP_SWEEP_T
    APs.append(new_AP)



# Generate a new location that is not too close to existing locations
def generate_sparse_location(existing_locations, min_dist=0.05): 
    max_attempts = 100
    for _ in range(max_attempts):
        pos = np.random.rand(2)
        pos[0] = 0.05 + 0.9 * pos[0]
        pos[1] = 0.05 + 0.9 * pos[1]
        if all(np.linalg.norm(pos - p) >= min_dist for p in existing_locations):
            return pos
    raise ValueError("Failed to generate a valid location")

# Generate food sources and ensure they are not too close to each other and APs
all_locations = [HIVE_POS_NORM] + AP_POS_NORM
food_sources = []
for _ in range(NUM_FOOD_SOURCES):
    pos = generate_sparse_location(all_locations)
    all_locations.append(pos)
    food_sources.append(Food(pos))

# Output the environment topology after generation
print ("\nEnvironment Topology: ")
print (f"├─ Field Size: {FIELD_SIZE}x{FIELD_SIZE/FIELD_ASPECT_RATIO:.1f} meters")
print (f"├─ AP locations (Total {AP_NUM}):")
for i, ap in enumerate(APs):
    print (f"│   {'├─' if i < AP_NUM-1 else '└─'} AP {i}: [{ap.location[0]:.3f}, {ap.location[1]:.3f}] m")
hive_pos_m = (
    HIVE_POS_NORM[0] * FIELD_SIZE,
    HIVE_POS_NORM[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
)
print (f"├─ Hive location: [{hive_pos_m[0]:.3f}, {hive_pos_m[1]:.3f}] m")
print (f"└─ Food Sources (Total {len(food_sources)}):")
for i, fs in enumerate(food_sources):
    fs_pos_m = (
        fs.location_normalized[0] * FIELD_SIZE,
        fs.location_normalized[1] * FIELD_SIZE / FIELD_ASPECT_RATIO
    )
    connector = '├─' if i < len(food_sources)-1 else '└─'
    print (f"    {connector} Source {i}: [{fs_pos_m[0]:.3f}, {fs_pos_m[1]:.3f}] m")

# Find the NLOS attenuation map and save it to the stats folder
PENETRATION_LOSS_HIVE = 15
PENETRATION_LOSS_AP = 10
PENETRATION_LOSS_FOOD = 5
PATH_OBSTACLE_DISTANCE = 1  # Along the path, 1m inside is considered penetrated
OBSCATLE_ATTENUATION_FACTOR = 50
FREESPACE_ATTENUATION_FACTOR = 20

# Plot the NLOS attenuation map
def plot_attenuation_map(ap_id, attenuation_grid): 
    '''
    Plot the NLOS attenuation map for a specific AP
    ap_id: the id of the AP
    attenuation_grid: the attenuation grid just obtained
    '''
    plt.figure(figsize=(10, 8))
    vmax = np.percentile(attenuation_grid, 95)
    img = plt.imshow(attenuation_grid.T, cmap='plasma', origin='lower', vmin=0, vmax=vmax,
                    extent=[0, FIELD_SIZE, 0, FIELD_SIZE/FIELD_ASPECT_RATIO])
    
    # Add the contour lines
    X, Y = np.meshgrid(np.arange(FIELD_SIZE), np.arange(int(FIELD_SIZE/FIELD_ASPECT_RATIO)))
    levels = np.linspace(0, vmax, 15 + 1)
    plt.contour(X, Y, attenuation_grid.T, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    
    # Adjust the colorbar and environment markers
    cbar = plt.colorbar(img, label='Total Attenuation (dB)')
    cbar.set_ticks(np.linspace(0, vmax, 10 + 1))
    cbar.outline.set_edgecolor('black')
    plt.setp(cbar.ax.yaxis.get_ticklines(), color='black')
    APs[ap_id].draw_plot_environment_markers()

    # Add the title, labels and legend
    plt.title(f"AP {ap_id} Signal Attenuation Map")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    
    # Save the figure
    output_dir = f'sim_figures/AP_{ap_id}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "attenuation_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Find the NLOS attenuation map for each AP
def find_nlos():
    '''
    NLOS attenuation calculation only consider free space loss and obstacle attenuation
    Future work: consider the multi-path effect more carefully
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
        
        # Generate grid (1m resolution) and debug points
        x_coords = np.arange(0, FIELD_SIZE, 1)
        y_coords = np.arange(0, FIELD_SIZE / FIELD_ASPECT_RATIO, 1)
        debug_points = [(x_coords[0], y_coords[0]), 
                       (x_coords[-1], y_coords[-1])]
        
        # Open the csv file and write the result
        with open(csv_path, 'w') as f:
            attenuation_grid = np.zeros((FIELD_SIZE, int(FIELD_SIZE/FIELD_ASPECT_RATIO))) # Format: x,y,total_attenuation_dB
            for x in x_coords:
                for y in y_coords:
                    current_pos = np.array([x, y])
                    total_attenuation = 0
                    
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
                        
                        # Calculate the distance between the obstacle and the AP-receiver line
                        ap_pos = ap.location
                        recv_pos = np.array([x, y])
                        vec_AP2Recv = recv_pos - ap_pos
                        vec_AP2Obs = obs_pos - ap_pos
                        
                        # Find the projection parameter t and judge whether the obstacle is in the LOS path (instead of the opposite side)
                        t = np.dot(vec_AP2Obs, vec_AP2Recv) / np.dot(vec_AP2Recv, vec_AP2Recv)
                        vec_AP2Recv_3d = np.append(vec_AP2Recv, 0)
                        vec_AP2Obs_3d = np.append(vec_AP2Obs, 0)
                        cross_product = np.cross(vec_AP2Recv_3d, vec_AP2Obs_3d)
                        distance = np.linalg.norm(cross_product) / np.linalg.norm(vec_AP2Recv)
                        if (0 <= t <= 1) and (distance < PATH_OBSTACLE_DISTANCE):
                            attenuation += loss
                    
                    # Calculate the total attenuation and print the debug information
                    total_attenuation = fs_loss + attenuation
                    debug_mode = (x, y) in debug_points
                    if debug_mode:
                        print (f"\nAP{ap.ap_id} attenuation map @ location norm.({round(x,2)},{round(y,2)})")
                        print (f"├─ LOS Vector: {los_vector}")
                        print (f"├─ LOS Distance: {los_distance:.2f}m")
                        print (f"├─ Free Space Loss: {fs_loss:.2f}dB")
                        print (f"├─ NLOS Attenuation: {attenuation:.2f}dB")
                        print (f"└─ Total Attenuation: {total_attenuation:.2f}dB")

                    # Save the result to the csv file
                    attenuation_grid[int(x), int(y)] = total_attenuation
                    f.write(f"{int(round(x))},{int(round(y))},{total_attenuation:.1f}\n")
            
            # Plot the attenuation map with the result in the end
            plot_attenuation_map(ap.ap_id, attenuation_grid)

# Load the NLOS attenuation map for each AP
find_nlos()
for ap in APs: 
    nlos_attenuation_path = os.path.join(os.path.dirname(__file__),"sim_stats", f"AP_{ap.ap_id}", 'nlos_attenuation.csv')
    if (not os.path.exists(nlos_attenuation_path)): find_nlos()
    ap.load_attenuation_map()

# Run the AP tests after initialization
for ap in APs: 
    print (f"\nTest AP{ap.ap_id} Signal Characteristics ...")
    ap.test_signal_characteristics()
    print (f"\nTest AP{ap.ap_id} Signal Characteristics at (0,0) ...")
    ap.test_signal_charactertistics_at_location((0.0, 0.0))
    print (f"\nTest AP{ap.ap_id} Beamforming Optimal Phases ...")
    ap.test_beamforming_best_theta()



## Start Simulation
# Draw the initial elements
sim = Simulation(food_sources)
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_facecolor('black')
bee_scatter = ax.scatter([], [], c='yellow', s=50, alpha=0.2, edgecolors='white', label='Bees')
ax.plot([], [], 'ro', markersize=12, alpha=0.8)
ax.plot(HIVE_POS_NORM[0], HIVE_POS_NORM[1], 'gs', markersize=15, label='Hive')
food_scatter = ax.scatter(
    [fs.location_normalized[0] for fs in food_sources],
    [fs.location_normalized[1] for fs in food_sources],
    c='#1E90FF', s=[fs.nectar / NECTAR_EXPECTED_STORAGE * NECTAR_DISPLAY_SIZE * 2 for fs in food_sources], edgecolors='white'
)
for i, ap in enumerate(AP_POS_NORM):
    label = 'AP' if i == 0 else None
    ax.plot(ap[0], ap[1], 'r^', markersize=15, alpha=0.8, label=label)
food_labels = [ax.text(fs.location_normalized[0], fs.location_normalized[1]+0.02, str(fs.nectar), 
                      ha='center', va='bottom', color='white', fontsize=8) for fs in food_sources]
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='yellow', markersize=10, label='Bees', alpha=0.7, markeredgewidth=0), 
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#1E90FF', markersize=10, label='Food', markeredgewidth=0),
    plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='green', markersize=10, label='Hive', markeredgewidth=0),
    plt.Line2D([0], [0], marker='^', color='none', markerfacecolor='red', markersize=10, label='AP', markeredgewidth=0)
]
ax.legend(handles=legend_elements, loc='upper right')



## Simulation functions
# Determine how many trail traces to save
current_frame_id = 0
trail_save_num = 10
prev_save_time = 0
if (NUM_FRAMES < 100): 
    trail_save_num = math.ceil(NUM_FRAMES / 10)
MOVEMENT_DISAPPEAR_TIME = math.ceil(SIM_TIME / trail_save_num)

# Update bee history and save the bee trail
def save_bee_trails(current_time):
    os.makedirs('sim_figures', exist_ok=True)
    os.makedirs('sim_stats', exist_ok=True)

    for i, bee in enumerate(sim.bees):
        figure_dir = f'sim_figures/bee_{i:02d}'
        os.makedirs(figure_dir, exist_ok=True)
        stats_dir = f'sim_stats/bee_{i:02d}'
        os.makedirs(stats_dir, exist_ok=True)
            
        # Create the bee trail figure
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
        for pos, timestamp in bee.history_location:
            age = current_time - timestamp
            alpha = max(0, 1 - age/MOVEMENT_DISAPPEAR_TIME)
            if alpha > 0:
                trail_locations.append(pos)
                trail_alphas.append(alpha)
        
        if trail_locations:
            ax_bee.scatter(
                [p[0] for p in trail_locations],
                [p[1] for p in trail_locations],
                c="yellow", s=20, alpha=trail_alphas, edgecolors='none'
            )

        # Draw the estimated location from the bee's perspective
        # -> The estimated location is obtained with the received signal from APs
        est_locations = []
        est_alphas = []
        for pos, ts in bee.history_location_norm_est:
            age = current_time - ts
            alpha = max(0, 1 - age/MOVEMENT_DISAPPEAR_TIME)
            if alpha > 0:
                est_locations.append(pos)
                est_alphas.append(alpha)
        
        if est_locations:
            ax_bee.scatter(
                [p[0] for p in est_locations],
                [p[1] for p in est_locations],
                c='cyan', s=25, alpha=est_alphas,
                edgecolors='none', label='Estimated'
            )

        # Add the legends
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='yellow', markersize=8, label='Actual Path'),
            plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='cyan', markersize=8, label='Estimated'),
            plt.Line2D([0], [0], marker='^', color='none', markerfacecolor='red', markersize=8, alpha=0.7, label='AP'), 
        ]
        ax_bee.legend(handles=legend_elements, loc='lower left', fontsize=7)

        # Save the trail trace figure
        # Note that each figure only contains one bee's trail for a given time period
        global prev_save_time
        plt.savefig(f'{figure_dir}/bee_{i:02d}_trail_{prev_save_time}ms-{current_time}ms.png', dpi=150, bbox_inches='tight')
        plt.close(fig_bee)

        # Save the CSV data
        csv_path = f'{stats_dir}/bee_{i:02d}_trail_{prev_save_time}ms-{current_time}ms.csv'
        with open(csv_path, 'w') as f: 
            header = ['timestamp', 'actual_x', 'actual_y', 'estimated_x', 'estimated_y']
            header += [f'ap{ap_id}_angle' for ap_id in range(AP_NUM)]
            f.write(','.join(header) + '\n')
            all_ts = set()
            actual_data = {ts: pos for pos, ts in bee.history_location}
            est_data = {ts: pos for pos, ts in bee.history_location_norm_est}
            angle_data = {}
            for ap_id, data in bee.history_ap_angles_est.items():
                for angle, ts in data:
                    angle_data[ts] = angle.item()
                    all_ts.add(ts)
            all_ts.update(actual_data.keys())
            all_ts.update(est_data.keys())
            sorted_ts = sorted(all_ts)
            for ts in sorted_ts: 
                actual = actual_data.get(ts, (None, None))
                estimated = est_data.get(ts, (None, None))
                angle = angle_data.get(ts, {})
                row = [
                    f'{ts}',
                    f"{actual[0]:.4f}" if actual[0] is not None else '',
                    f"{actual[1]:.4f}" if actual[1] is not None else '',
                    f"{estimated[0]:.4f}" if estimated[0] is not None else '',
                    f"{estimated[1]:.4f}" if estimated[1] is not None else '',
                    f"{angle:.2f}" if f'{angle}'.isnumeric() else ''
                ]
                f.write(','.join(row) + '\n')

        # Draw the angle comparison figure
        # only for the first bee
        if (i == 0):
            for ap_id in range(AP_NUM):
                ap_dir = f'sim_figures/AP_{ap_id}'
                os.makedirs(ap_dir, exist_ok=True)
                
                # Collect the estimated angles from the bee's perspective
                est_angles = []
                est_timestamps = []
                actual_angles = []
                actual_timestamps = []
                if ap_id in bee.history_ap_angles_est:
                    for angle, ts in bee.history_ap_angles_est[ap_id]:
                        if prev_save_time <= ts <= current_time:
                            est_angles.append(angle)  # already in degrees
                            est_timestamps.append(ts)
                
                # Calculate the actual angle (based on the true position)
                for pos, ts in bee.history_location:
                    if prev_save_time <= ts <= current_time:
                        ap_pos = AP_POS_NORM[ap_id]
                        dx = pos[0] - ap_pos[0]
                        dy = pos[1] - ap_pos[1]
                        world_angle = np.arctan2(dy, dx) % np.pi  # 0-π范围
                        antenna_angle = AP_ANTENNA_LAYOUT_DIRECTION[ap_id]  # 0-π范围
                        actual_angles.append((np.degrees(world_angle - antenna_angle) + 360) % 180) # consider the antenna layout direction
                        actual_timestamps.append(ts)
                
                # Draw the angle comparison figure
                plt.figure(figsize=(10, 5))
                plt.plot(est_timestamps, est_angles, 'b-', label='Estimated')
                plt.plot(actual_timestamps, actual_angles, 'g--', label='Actual')
                plt.ylim(0, 180)
                plt.title(f'AP{ap_id} Angle Comparison ({prev_save_time}-{current_time}ms) for Bee {i}')
                plt.xlabel('Time (ms)')
                plt.ylabel('Angle (degrees)')
                plt.legend()
                plt.savefig(f'{ap_dir}/bee_{i:02d}_trail_angle_est_{prev_save_time}ms-{current_time}ms.png', dpi=150)
                plt.close()

# Last sample time of each AP
last_AP_sample_time = [ap.signal_start for ap in APs]

# Update the frame
def update(frame):
    global current_frame_id
    current_frame_id = frame
    current_time = current_frame_id * FRAME_TIME

    # AP ransmit signals to the bees
    transmitted = True
    while transmitted: 
        transmitted = False
        for ap in APs: 
            while current_time - last_AP_sample_time[ap.ap_id] > AP_SWEEP_T: 
                ap.transmit_signal(sim.bees, last_AP_sample_time[ap.ap_id])
                last_AP_sample_time[ap.ap_id] += AP_SWEEP_T * AP_NUM
                transmitted = True

    # Bees move and save their new locations
    for bee in sim.bees:
        bee.update_location_history(current_frame_id * FRAME_TIME)
    
    # Save the bee trail figure with the given interval
    if (frame + 1) % (NUM_FRAMES // trail_save_num) == 0: 
        print (f"Saving the bee trail figures (set {(frame + 1) // (NUM_FRAMES // trail_save_num)}) ...")
        save_bee_trails(current_time)
        global prev_save_time
        prev_save_time = current_time + FRAME_TIME
    
    # Update the bees and nectars (and their scattar displays)
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

    return [bee_scatter, food_scatter] + food_labels

# Save the animation when it finishes
def print_progress(i, n):
    print (f'\nProgress: {i+1}/{n} frames', end='\r')
    if (i == n - 1): 
        print (f'\nPlease wait for a while for the amination to save ...')

print ("\nGenerating simulation amination ...")
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=FRAME_TIME, blit=True, repeat=False)
ani.save(os.path.join(os.path.dirname(__file__), 'sim_figures', 'sim.gif'), writer='pillow', fps=1000//FRAME_TIME,
         progress_callback=lambda i, n: print_progress(i, n))
print ("\nAnimation saved! ")
plt.close('all')
os._exit(0) 
