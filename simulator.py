"""
蜜蜂飞行轨迹模拟 - 人工蜂群算法优化
依赖库：numpy, matplotlib
"""
import numpy as np
import os, datetime, math, shutil, random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Environment and Display Parameters
NUM_BEES = 10
NUM_FOOD_SOURCES = 5  
HIVE_POS = (0.5, 0.1)                  # normalized
AP_POS = [(0.25, 0.5), (0.75, 0.5)]    # normalized
FIG_SIZE = (8, 8)
NECTAR_DISPLAY_SIZE = 50
NECTAR_EXPECTED_STORAGE = 15
NECTAR_EXPECTED_RECOVERY_TIME = 20

# Simulation Parameters
FRAME_TIME = 50    # ms
SIM_TIME = 6000   # ms
NUM_FRAMES = round(SIM_TIME / FRAME_TIME)

# Bee Parameters
# bumblebee movement speed: x m/s (normally)
# which is (x / FIELD_SIZE) * normalized_edge_length m/s
# 1s = 20 frames
# 1 frame = (x / FIELD_SIZE / 20) * normalized_edge_length m
FIELD_SIZE = 50        # m
BUMBLEBEE_SPEED = 5    # m/s
STEP_SIZE = BUMBLEBEE_SPEED / FIELD_SIZE / 20
MIN_STEP = 0.75 * STEP_SIZE 
MAX_STEP = 1.25 * STEP_SIZE
STEP_NOISE = 0.25 * STEP_SIZE
NECTAR_DISCOVERY_DISTANCE = 5 * STEP_SIZE
NECTAR_EMPTY_DISCOVERY_DISTANCE = 10 * STEP_SIZE
OVERLAP_DISTANCE_ERROR = 3 * STEP_SIZE
MOVEMENT_DISAPPEAR_TIME = 10000 # ms


if os.path.exists('bee_flight.gif'):
    os.remove('bee_flight.gif')
if os.path.exists('sim_figures'):
    shutil.rmtree('sim_figures')

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
        self.known_empty_sources = set()  # 只记录自己发现的无蜜蜜源
        self.history = []  # 新增：记录位置和时间的列表 (pos, timestamp)

    def reset_knowledge(self):
        self.known_empty_sources.clear()

    def update_history(self, frame_time):
        # 添加当前位置和时间戳
        self.history.append((self.position.copy(), frame_time))
        # 保持历史记录长度
        if len(self.history) > math.ceil(MOVEMENT_DISAPPEAR_TIME / FRAME_TIME):
            self.history.pop(0)

class Simulation:
    def __init__(self, food_sources):
        self.start_time = datetime.datetime.now()
        self.bees = [Bee(food_sources) for _ in range(NUM_BEES)]
        self.food_sources = food_sources
        self.frame_count = 0

    def get_current_time(self):
        return (datetime.datetime.now() - self.start_time).total_seconds() * 1000

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

# Generate food sources
def generate_valid_position(existing_positions, min_dist=0.1):
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
def update(frame):
    
    # Update bee history and save the bee trail
    def save_bee_trails(frame, current_time):
        os.makedirs('sim_figures', exist_ok=True)
        for i, bee in enumerate(sim.bees):

            # Create a unique folder for each bee
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
            
            # Save and close
            plt.savefig(f'{bee_dir}/trail_{frame:04d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig_bee)
    current_time = sim.get_current_time()
    for bee in sim.bees:
        bee.update_history(current_time)
    if (frame + 1) % (NUM_FRAMES // 10) == 0:
        save_bee_trails(frame+1, current_time)
    
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
print("Generating animation...")
ani = FuncAnimation(
    fig, 
    update, 
    frames=NUM_FRAMES, 
    interval=FRAME_TIME, 
    blit=True,
    repeat=False
)
ani.save('bee_flight.gif', 
         writer='pillow', 
         fps=1000//FRAME_TIME,
         progress_callback=lambda i, n: print(f'Progress: {i+1}/{n} frames', end='\r'))
print("\nAnimation saved! ")
