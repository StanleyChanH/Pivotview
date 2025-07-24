# PivotView/main.py
#
# 作者: StanleyChanH
# 版本: 2.0.0
# 描述: 本脚本是 PivotView 项目的主控制器。
#      [V2.0] 引入有限状态机 (FSM) 重构核心逻辑，
#      实现了智能化的目标丢失策略 (螺旋搜索) 和平滑的目标捕获模式，
#      极大提升了系统的鲁棒性和跟踪体验的优雅性。

import time
import math
import threading
from enum import Enum, auto

# --- 导入子项目中的模块 ---
from receivemaix.main import MaixCamReceiver
from ServoController.servo_controller_library import ServoController, ServoGroup, Easing, ServoHandle

# =============================================================================
# --- 1. 全局配置 ---
# =============================================================================
# MaixCam 摄像头输出的图像分辨率
CAM_WIDTH = 224
CAM_HEIGHT = 224

# 串口配置
SERIAL_PORT = '/dev/ttyAMA0'
BAUDRATE = 115200

# 舵机驱动板 I2C 地址
SERVO_BOARD_ADDRESS = 0x2D

# 舵机通道号 (Pan/Tilt)
H_SERVO_CHANNEL = 1
V_SERVO_CHANNEL = 2

# 舵机物理规格
H_SERVO_MAX_ANGLE = 270
V_SERVO_MAX_ANGLE = 180

# 舵机安全活动范围 (0-100的百分比)
H_SERVO_MIN_SAFE_PERCENT = 10.0
H_SERVO_MAX_SAFE_PERCENT = 90.0
V_SERVO_MIN_SAFE_PERCENT = 10.0
V_SERVO_MAX_SAFE_PERCENT = 90.0

# [V2.0] 状态机与行为策略配置
class State(Enum):
    """有限状态机的状态定义"""
    IDLE = auto()      # 空闲/归中状态
    ACQUIRING = auto() # 捕获目标中 (平滑移向中心)
    TRACKING = auto()  # 正在跟踪目标
    SEARCHING = auto() # 目标丢失，正在搜索

TARGET_LOST_TIMEOUT = 1.0       # 从TRACKING切换到SEARCHING的超时(秒)
SEARCH_TIMEOUT = 5.0            # 从SEARCHING切换到IDLE的超时(秒)
ACQUIRE_KP = 0.6                # [V2.0] 平滑捕获模式下的P增益
SEARCH_SPIRAL_SPEED = 1.0       # [V2.0] 螺旋搜索的速度
SEARCH_SPIRAL_RADIUS_SPEED = 5.0 # [V2.0] 螺旋搜索的半径增长速度

# PID 高级配置
CENTER_DEAD_ZONE = 0.03  # 中心死区范围 (3%)
ADAPTIVE_GAIN_ENABLED = True

# =============================================================================
# --- 2. PID 控制器 ---
# (与v1.3.0版本相同，保持其优秀性能)
# =============================================================================
class PIDController:
    """一个通用的PID控制器，支持积分抗饱和、中心死区和动态增益。"""
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(-100, 100), dead_zone=0.0):
        self.base_Kp = Kp
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.dead_zone = dead_zone
        self._integral = 0
        self.last_time = time.time()
        self.last_error = 0

    def update(self, process_variable):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return 0
        error = self.setpoint - process_variable
        if abs(error) < self.dead_zone:
            self.reset_integral()
            return 0
        self._integral += self.Ki * error * dt
        self._integral = max(min(self._integral, self.output_limits[1]), self.output_limits[0])
        derivative = self.Kd * (error - self.last_error) / dt
        output = self.Kp * error + self._integral + derivative
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        self.last_error = error
        self.last_time = current_time
        return output
    
    def set_kp(self, new_kp):
        self.Kp = new_kp
        
    def reset_integral(self):
        self._integral = 0

    def reset(self):
        self.Kp = self.base_Kp
        self._integral = 0
        self.last_error = 0
        self.last_time = time.time()

# =============================================================================
# --- 3. 状态处理函数 [V2.0] ---
# =============================================================================
def handle_idle_state(gimbal, initial_pose):
    """处理IDLE状态：让云台缓慢归中。"""
    print("状态: IDLE (空闲/归中)")
    gimbal.move_to_pose(initial_pose, duration=2.0, easing_func=Easing.ease_in_out_cubic)
    gimbal.wait_for_move() # 等待归中完成
    return State.IDLE # 保持IDLE状态，等待目标

def handle_acquiring_state(gimbal, target_data, h_servo, v_servo):
    """处理ACQUIRING状态：平滑将目标移至中心。"""
    target_center_x = (target_data['x'] + target_data['w'] / 2) / CAM_WIDTH
    target_center_y = (target_data['y'] + target_data['h'] / 2) / CAM_HEIGHT

    error_x = 0.5 - target_center_x
    error_y = 0.5 - target_center_y

    # 如果已经接近中心，则切换到TRACKING状态
    if abs(error_x) < CENTER_DEAD_ZONE and abs(error_y) < CENTER_DEAD_ZONE:
        print("目标已捕获，切换到TRACKING模式...")
        return State.TRACKING

    # 使用简单的P控制器进行平滑捕获
    h_adjustment = ACQUIRE_KP * error_x * 10 # 乘以10以获得合适的调整量
    v_adjustment = ACQUIRE_KP * error_y * 10
    
    new_h_pos = h_servo.current_position + h_adjustment
    new_v_pos = v_servo.current_position - v_adjustment # 垂直方向可能反向

    gimbal.move_to_pose(
        {'h': new_h_pos, 'v': new_v_pos},
        duration=0.2,
        easing_func=Easing.ease_in_out_quad
    )
    print(f"状态: ACQUIRING (捕获中)... Error:({error_x:.2f}, {error_y:.2f})", end='\r')
    return State.ACQUIRING

def handle_tracking_state(gimbal, target_data, pid_h, pid_v, h_servo, v_servo, base_pid_h, base_pid_v):
    """处理TRACKING状态：高性能PID跟踪。"""
    if ADAPTIVE_GAIN_ENABLED:
        normalized_w = (target_data['w'] / CAM_WIDTH) + 1e-5
        gain_factor = min(2.0, max(0.5, 1.0 / normalized_w))
        pid_h.set_kp(base_pid_h['Kp'] * gain_factor)
        pid_v.set_kp(base_pid_v['Kp'] * gain_factor)

    target_center_x = (target_data['x'] + target_data['w'] / 2) / CAM_WIDTH
    target_center_y = (target_data['y'] + target_data['h'] / 2) / CAM_HEIGHT

    h_adjustment = pid_h.update(target_center_x)
    v_adjustment = pid_v.update(target_center_y)
    
    if h_adjustment != 0 or v_adjustment != 0:
        new_h_pos = h_servo.current_position - h_adjustment
        new_v_pos = v_servo.current_position + v_adjustment
        gimbal.move_to_pose(
            {'h': new_h_pos, 'v': new_v_pos},
            duration=0.1, 
            easing_func=Easing.ease_in_out_cubic
        )
    print(f"状态: TRACKING (跟踪中)... H:{h_servo.current_position:.1f}% V:{v_servo.current_position:.1f}%", end='\r')
    return State.TRACKING

def handle_searching_state(gimbal, search_params):
    """处理SEARCHING状态：螺旋搜索目标。"""
    now = time.time()
    dt = now - search_params['last_update_time']

    search_params['angle'] += SEARCH_SPIRAL_SPEED * dt
    search_params['radius'] += SEARCH_SPIRAL_RADIUS_SPEED * dt
    
    # 限制最大搜索半径
    search_params['radius'] = min(search_params['radius'], 40) 

    offset_h = search_params['radius'] * math.cos(search_params['angle'])
    offset_v = search_params['radius'] * math.sin(search_params['angle'])
    
    target_h = search_params['last_pos']['h'] + offset_h
    target_v = search_params['last_pos']['v'] + offset_v

    gimbal.move_to_pose({'h': target_h, 'v': target_v}, duration=0.2, easing_func=Easing.linear)
    search_params['last_update_time'] = now
    
    print(f"状态: SEARCHING (螺旋搜索中)... R:{search_params['radius']:.1f}", end='\r')
    return State.SEARCHING

# =============================================================================
# --- 4. 主程序 ---
# =============================================================================
def main():
    print("--- [PivotView v2.0] 智能行为跟踪系统启动 ---")

    # --- 初始化硬件 ---
    print("\n[1/3] 初始化舵机系统...")
    try:
        servo_controller = ServoController(board_address=SERVO_BOARD_ADDRESS)
        if not servo_controller.bus: raise ConnectionError("无法连接舵机驱动板")
        h_servo = servo_controller.setup_servo(
            H_SERVO_CHANNEL, max_angle=H_SERVO_MAX_ANGLE,
            min_safe_percent=H_SERVO_MIN_SAFE_PERCENT, max_safe_percent=H_SERVO_MAX_SAFE_PERCENT)
        v_servo = servo_controller.setup_servo(
            V_SERVO_CHANNEL, max_angle=V_SERVO_MAX_ANGLE,
            min_safe_percent=V_SERVO_MIN_SAFE_PERCENT, max_safe_percent=V_SERVO_MAX_SAFE_PERCENT)
        gimbal = ServoGroup(h=h_servo, v=v_servo)
        servo_controller.run_initialization_sequence(duration_per_move=1.5)
        initial_pose = {'h': 50, 'v': 50}
    except Exception as e:
        print(f"[致命错误] 舵机系统初始化失败: {e}"); return

    print("\n[2/3] 初始化串口接收器...")
    receiver = MaixCamReceiver(port=SERIAL_PORT, baudrate=BAUDRATE, debug=False)
    if not receiver.start():
        print("[致命错误] 无法启动串口接收器"); servo_controller.cleanup(); return
    
    print("\n[3/3] 初始化PID与状态机...")
    base_pid_h = {'Kp': 0.4, 'Ki': 0.02, 'Kd': 0.05}
    base_pid_v = {'Kp': 0.5, 'Ki': 0.03, 'Kd': 0.06}
    pid_h = PIDController(**base_pid_h, setpoint=0.5, output_limits=(-5, 5), dead_zone=CENTER_DEAD_ZONE)
    pid_v = PIDController(**base_pid_v, setpoint=0.5, output_limits=(-5, 5), dead_zone=CENTER_DEAD_ZONE)
    
    # --- 主循环 (FSM) ---
    current_state = State.IDLE
    last_target_time = 0
    search_params = {}
    
    print("\n--- 进入主循环 (按 Ctrl+C 退出) ---")
    try:
        while True:
            target_data = receiver.get_latest_data()

            # --- 状态转换逻辑 ---
            if target_data:
                last_target_time = time.time()
                if current_state in [State.IDLE, State.SEARCHING]:
                    current_state = State.ACQUIRING
                elif current_state == State.ACQUIRING:
                    # ACQUIRING内部会判断是否切换到TRACKING
                    pass
                # 如果是TRACKING，则保持
            else: # 无目标数据
                if current_state == State.TRACKING and time.time() - last_target_time > TARGET_LOST_TIMEOUT:
                    print("\n目标丢失，切换到SEARCHING模式...")
                    current_state = State.SEARCHING
                    # 初始化搜索参数
                    search_params = {
                        'start_time': time.time(),
                        'last_update_time': time.time(),
                        'last_pos': {'h': h_servo.current_position, 'v': v_servo.current_position},
                        'angle': 0,
                        'radius': 0
                    }
                elif current_state == State.SEARCHING and time.time() - search_params['start_time'] > SEARCH_TIMEOUT:
                    print("\n搜索超时，切换到IDLE模式...")
                    current_state = State.IDLE
            
            # --- 状态执行逻辑 ---
            if current_state == State.IDLE:
                # IDLE状态只在初次进入或搜索超时后执行一次归中，然后等待
                if 'executed' not in locals() or not executed:
                    handle_idle_state(gimbal, initial_pose)
                    executed = True # 标记已执行，避免循环归中
                time.sleep(0.1) # 空闲时降低CPU占用
                continue # 等待下一次循环
            
            executed = False # 如果离开IDLE状态，重置执行标记

            if current_state == State.ACQUIRING:
                current_state = handle_acquiring_state(gimbal, target_data, h_servo, v_servo)
            elif current_state == State.TRACKING:
                current_state = handle_tracking_state(gimbal, target_data, pid_h, pid_v, h_servo, v_servo, base_pid_h, base_pid_v)
            elif current_state == State.SEARCHING:
                current_state = handle_searching_state(gimbal, search_params)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    finally:
        print("正在执行清理操作...")
        receiver.stop()
        print("    > 云台最终归中...")
        gimbal.move_to_pose(initial_pose, duration=1.5).wait_for_move()
        servo_controller.cleanup()
        print("清理完成。程序安全退出。")

if __name__ == "__main__":
    main()
