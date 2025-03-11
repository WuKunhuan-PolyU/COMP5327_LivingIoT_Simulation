"""
蜜蜂飞行轨迹模拟 - 人工蜂群算法优化
依赖库：numpy, matplotlib
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import os
import datetime
import math
import shutil
matplotlib.use('Agg')  # 使用非交互式后端

# ABC 算法参数
NUM_BEES = 10           # 减少蜜蜂数量
LIMIT = 100            # 放弃蜜源阈值
SEARCH_RANGE = (0, 1)   # 调整搜索范围为0-1
HIVE_POS = (0.5, 0.1)   # 蜂巢位置
NUM_FOOD_SOURCES = 5    # 蜜源数量
AP_POSITIONS = [(0.25, 0.5), (0.75, 0.5)]  # 访问点位置
NECTAR_FULL = 15
NECTAR_RECOVERY_TIME = 8

# 模拟参数
FIG_SIZE = (8, 8)      # 画布尺寸
FRAME_TIME = 50      # 飞行速度（毫秒/帧）
SIM_TIME = 60000      # 动画时长（毫秒）
GIF_FRAMES = round(SIM_TIME / FRAME_TIME)
FLIGHT_NOISE = 0.01
MIN_STEP = 0.01
MAX_STEP = 0.02
BASE_FOOD_SIZE = 50

# 记录参数
BEE_MOVEMENT_DISAPPEAR_TIME = 10000 # 蜜蜂移动消失时间 (毫秒)
BEE_HISTORY_LENGTH = 200  # 每个蜜蜂记录的历史位置数

class FoodSource:
    def __init__(self, position):
        self.position = position
        self.nectar = round((0.8 + 0.4 * random.random()) * NECTAR_FULL)
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
        if len(self.history) > math.ceil(BEE_MOVEMENT_DISAPPEAR_TIME / FRAME_TIME):
            self.history.pop(0)

# 修改蜜源初始化部分
def generate_valid_position(existing_positions, min_dist=0.1):
    while True:
        pos = np.random.rand(2)
        pos[0] = 0.05 + 0.9 * pos[0]
        pos[1] = 0.05 + 0.9 * pos[1]
        if all(np.linalg.norm(pos - p) >= min_dist for p in existing_positions):
            return pos

# 生成蜜源时考虑最小距离
all_positions = [HIVE_POS] + AP_POSITIONS
food_sources = []
for _ in range(NUM_FOOD_SOURCES):
    pos = generate_valid_position(all_positions)
    all_positions.append(pos)
    food_sources.append(FoodSource(pos))

# 目标函数（示例：Rastrigin函数）
def fitness_function(pos):
    return -((pos[0]**2 - 10*np.cos(2*np.pi*pos[0])) + 
            (pos[1]**2 - 10*np.cos(2*np.pi*pos[1])) + 20)

class ABCAlgorithm:
    def __init__(self, food_sources):
        self.start_time = datetime.datetime.now()
        self.bees = [Bee(food_sources) for _ in range(NUM_BEES)]
        self.food_sources = food_sources
        self.frame_count = 0

    def get_current_time(self):
        return (datetime.datetime.now() - self.start_time).total_seconds() * 1000  # 毫秒

    def update_nectar(self):
        self.frame_count += 1
        for i, fs in enumerate(self.food_sources):
            if fs.nectar == -1: 
                if fs.recovery_timer > 0:
                    fs.recovery_timer -= 1
                else:
                    fs.nectar = round((0.8 + 0.4 * random.random()) * NECTAR_FULL)
                    fs.recovery_timer = 0

    def update_positions(self):
        self.update_nectar()
        
        for bee in self.bees:
            if not bee.has_food:
                # 在移动前检查附近蜜源
                for fs in self.food_sources:
                    distance = np.linalg.norm(fs.position - bee.position)
                    if distance < 0.1:
                        if fs.nectar <= 0:
                            bee.known_empty_sources.add(fs)
                        else:
                            bee.target_food = fs
                            bee.known_empty_sources.discard(fs)
                    
                    # 规则2：中距离(0.2)吸引力判断
                    if (distance < 0.2 and 
                        fs != bee.target_food and 
                        fs.nectar > 0 and 
                        fs not in bee.known_empty_sources):
                        
                        # 比较当前目标距离
                        if bee.target_food is None:
                            bee.target_food = fs
                        else:
                            current_dist = np.linalg.norm(bee.target_food.position - bee.position)
                            if distance < current_dist:
                                bee.target_food = fs
                
                # 如果当前目标被发现无蜜，立即清除
                if (bee.target_food and 
                    bee.target_food in bee.known_empty_sources):
                    bee.target_food = None
                
                # 自动寻找新目标
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
                
                if bee.target_food:
                    direction = bee.target_food.position - bee.position
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        step_size = np.clip(0.02 * distance, MIN_STEP, MAX_STEP)
                        step = (direction / distance) * step_size
                        bee.position += step + np.random.normal(0, FLIGHT_NOISE, 2)
                    
                    # 到达后检查蜜量
                    if np.linalg.norm(bee.position - bee.target_food.position) < 0.02:
                        if bee.target_food.nectar > 0:
                            bee.target_food.nectar -= 1
                            bee.has_food = True
                        else:
                            bee.known_empty_sources.add(bee.target_food)
                        bee.target_food = None
            else:
                # 返回蜂巢
                direction = np.array(HIVE_POS) - bee.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    step_size = np.clip(0.02 * distance, MIN_STEP, MAX_STEP)
                    step = (direction / distance) * step_size  # 标准化方向向量
                    bee.position += step + np.random.normal(0, FLIGHT_NOISE, 2)
                
                # 到达蜂巢判断
                if np.linalg.norm(bee.position - HIVE_POS) < 0.02:
                    bee.has_food = False

        for i, fs in enumerate(self.food_sources):
            if fs.nectar == 0: 
                fs.nectar = -1
                fs.recovery_timer = 1000 / FRAME_TIME * NECTAR_RECOVERY_TIME


# 在初始化算法和画布前添加清理代码
if os.path.exists('bee_flight.gif'):
    os.remove('bee_flight.gif')
if os.path.exists('sim_figures'):
    shutil.rmtree('sim_figures')

# 初始化算法和画布
abc = ABCAlgorithm(food_sources)
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_facecolor('black')

# 初始化蜜蜂图形元素
scatter = ax.scatter(
    [], [], 
    c='yellow', 
    s=50, 
    alpha=0.2,
    edgecolors='white',
    label='Bees'  # 添加标签
)
best_bee, = ax.plot([], [], 'ro', markersize=12, alpha=0.8)

# 恢复主图形的元素绘制
# 蜂巢（绿色方块）
ax.plot(HIVE_POS[0], HIVE_POS[1], 'gs', markersize=15, label='Hive')

# 蜜源（蓝色圆点）
food_scatter = ax.scatter(
    [fs.position[0] for fs in food_sources],
    [fs.position[1] for fs in food_sources],
    c='#1E90FF', 
    s=[fs.nectar/NECTAR_FULL*BASE_FOOD_SIZE*2 for fs in food_sources],
    edgecolors='white'
)

# 访问点（红色三角形）
for i, ap in enumerate(AP_POSITIONS):
    label = 'AP' if i == 0 else None  # 只为第一个AP添加标签
    ax.plot(ap[0], ap[1], 'r^', markersize=15, alpha=0.8, label=label)

# 恢复蜜量文字标签的更新
food_labels = [ax.text(fs.position[0], fs.position[1]+0.02, str(fs.nectar), 
                      ha='center', va='bottom', color='white', fontsize=8)
              for fs in food_sources]

# 修改图例元素设置
legend_elements = [
    plt.Line2D([0], [0], 
               marker='o', 
               color='none',  # 透明线条
               markerfacecolor='yellow',
               markersize=10, 
               label='Bees',
               alpha=0.7,
               markeredgewidth=0),  # 移除边缘
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

# 修改轨迹保存部分
def save_bee_trails(frame, current_time):
    os.makedirs('sim_figures', exist_ok=True)
    
    for i, bee in enumerate(abc.bees):
        # 创建每个蜜蜂的专属文件夹
        bee_dir = f'sim_figures/bee_{i:02d}'
        os.makedirs(bee_dir, exist_ok=True)
        
        # 创建新画布
        fig_bee, ax_bee = plt.subplots(figsize=FIG_SIZE)
        ax_bee.set_xlim(0, 1)
        ax_bee.set_ylim(0, 1)
        ax_bee.set_facecolor('black')
        
        # 只绘制访问点
        for ap in AP_POSITIONS:
            ax_bee.plot(ap[0], ap[1], 'r^', markersize=15, alpha=0.8)
        
        # 绘制该蜜蜂的轨迹
        trail_positions = []
        trail_alphas = []
        for pos, timestamp in bee.history:
            age = current_time - timestamp
            alpha = max(0, 1 - age/BEE_MOVEMENT_DISAPPEAR_TIME)
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
        
        # 保存并关闭
        plt.savefig(f'{bee_dir}/trail_{frame:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_bee)



def update(frame):
    current_time = abc.get_current_time()
    
    # 更新所有蜜蜂的历史记录
    for bee in abc.bees:
        bee.update_history(current_time)
    
    # 保存一次轨迹图（共保存10次）
    if frame % (GIF_FRAMES//10) == 0:
        save_bee_trails(frame, current_time)
    
    abc.update_positions()
    
    # 更新蜜蜂位置和透明度
    positions = np.array([b.position for b in abc.bees])
    alphas = [0.7 if b.has_food else 0.2 for b in abc.bees]
    scatter.set_offsets(positions)
    scatter.set_alpha(alphas)
    
    # 更新蜜源状态
    food_sizes = []
    food_alphas = []
    for fs in abc.food_sources:
        if fs.nectar > 0:
            size = (1 + (fs.nectar/NECTAR_FULL)) * BASE_FOOD_SIZE
            alpha = 0.7
        else:
            size = BASE_FOOD_SIZE
            alpha = 0.2
        food_sizes.append(size)
        food_alphas.append(alpha)
    food_scatter.set_sizes(food_sizes)
    food_scatter.set_alpha(food_alphas)

    # 恢复蜜量标签更新
    for fs, label in zip(abc.food_sources, food_labels):
        if fs.nectar > 0: 
            label.set_text(str(fs.nectar))
            label.set_visible(True)
        else:
            label.set_visible(False)
    
    return [scatter, food_scatter] + food_labels

# 创建动画
ani = FuncAnimation(
    fig, 
    update, 
    frames=GIF_FRAMES, 
    interval=FRAME_TIME, 
    blit=True,
    repeat=False
)

# 保存GIF并自动关闭窗口
print("正在生成动画...")
ani.save('bee_flight.gif', 
         writer='pillow', 
         fps=1000//FRAME_TIME,  # 根据帧间隔计算正确fps
         progress_callback=lambda i, n: print(f'进度: {i+1}/{n} 帧', end='\r'))
print("\n动画保存完成!")
plt.close(fig)
plt.show()

# 在文件最后添加（保存完成后打开文件夹）
print("正在打开结果文件夹...")

