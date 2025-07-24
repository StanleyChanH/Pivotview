# PivotView/main.py
#
# 作者: StanleyChanH
# 版本: 3.0.0
# 描述: 本脚本是 PivotView 项目的主控制器，在树莓派上运行。
#      此版本引入了专业的日志系统，并基于有限状态机 (FSM) 实现高级行为策略，
#      包括平滑捕获、螺旋搜索、自适应增益PID等，以实现高性能的视觉追踪。

import time
import math
import logging
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

# 舵机通道号
H_SERVO_CHANNEL = 1
V_SERVO_CHANNEL = 2
H_SERVO_MAX_ANGLE = 270
V_SERVO_MAX_ANGLE = 180

# 舵机安全活动范围 (0-100的百分比)
H_SERVO_MIN_SAFE_PERCENT = 0.0
H_SERVO_MAX_SAFE_PERCENT = 100.0
V_SERVO_MIN_SAFE_PERCENT = 10.0
V_SERVO_MAX_SAFE_PERCENT = 90.0

# --- 行为策略配置 ---
TARGET_LOST_TIMEOUT = 1.0       # 目标丢失后，从此时间开始进入搜索状态 (秒)
SEARCH_TIMEOUT = 10.0           # 搜索状态持续时间，超时后返回IDLE (秒)
SEARCH_SPIRAL_DENSITY = 10      # 螺旋搜索的密度，值越大点越密
SEARCH_SPIRAL_MAX_RADIUS = 0.4  # 螺旋搜索的最大半径 (标准化坐标)
ACQUIRE_KP = 0.6                # 捕获模式下的P控制器增益
ACQUIRE_THRESHOLD = 0.05        # 进入此误差范围，视为捕获成功，切换到TRACKING

# --- PID与控制配置 ---
CENTER_DEAD_ZONE = 0.02         # 中心死区范围 (标准化坐标)
ADAPTIVE_GAIN_ENABLED = True    # 是否启用Z轴自适应增益
# 基础PID参数 (将在自适应增益中动态调整)
# Kp: 比例, Ki: 积分, Kd: 微分
base_pid_h = {'Kp': 0.4, 'Ki': 0.02, 'Kd': 0.05, 'setpoint': 0.5}
base_pid_v = {'Kp': 0.5, 'Ki': 0.03, 'Kd': 0.06, 'setpoint': 0.5}


# =============================================================================
# --- 2. 日志系统配置 ([优化] 引入日志系统) ---
# =============================================================================
def setup_logging():
    """配置日志系统，实现控制台和文件双路输出"""
    logger = logging.getLogger("PivotView")
    logger.setLevel(logging.DEBUG)

    # 防止重复添加handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 文件处理器 - 记录所有DEBUG级别及以上的信息
    fh = logging.FileHandler('pivotview.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台处理器 - 只显示INFO级别及以上的信息
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# 初始化logger
logger = setup_logging()


# =============================================================================
# --- 3. PID 控制器 ---
# =============================================================================
class PIDController:
    """一个通用的、带高级功能的PID控制器"""
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(-5, 5), dead_zone=0.0):
        self.base_Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Kp = Kp
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.dead_zone = dead_zone

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self.last_time = time.time()
        self.last_error = 0

    def update(self, process_variable):
        """计算PID输出值"""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return 0

        error = self.setpoint - process_variable

        # [优化] 中心死区
        if abs(error) < self.dead_zone:
            self.reset_integral() # 在死区内时重置积分，防止漂移
            return 0.0

        self._proportional = self.Kp * error
        
        # [优化] 积分抗饱和
        self._integral += self.Ki * error * dt
        self._integral = max(min(self._integral, self.output_limits[1]), self.output_limits[0])

        self._derivative = self.Kd * (error - self.last_error) / dt

        output = self._proportional + self._integral + self._derivative
        output = max(min(output, self.output_limits[1]), self.output_limits[0])

        self.last_error = error
        self.last_time = current_time
        return output

    def reset(self):
        """重置PID控制器状态"""
        self._integral = 0
        self.last_error = 0
        self.last_time = time.time()
        self.Kp = self.base_Kp # [优化] 重置动态调整过的Kp

    def reset_integral(self):
        self._integral = 0

    def set_kp(self, new_kp): # [优化] 用于自适应增益
        self.Kp = new_kp


# =============================================================================
# --- 4. 状态机与主程序 ---
# =============================================================================
class State(Enum):
    """定义云台的有限状态"""
    IDLE = auto()       # 空闲/归位状态
    ACQUIRING = auto()  # 捕获新目标状态
    TRACKING = auto()   # 稳定跟踪状态
    SEARCHING = auto()  # 搜索丢失目标状态


def get_spiral_point(t, max_radius, density):
    """生成螺旋线上的点"""
    angle = t * density
    radius = t * max_radius
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    return x, y

def main():
    """主程序入口"""
    logger.info("--- [PivotView] v3.0 智能视觉追踪系统启动 ---")

    # --- 初始化舵机系统 ---
    logger.info("[1/4] 初始化舵机系统...")
    try:
        controller = ServoController(board_address=SERVO_BOARD_ADDRESS)
        if not controller.bus:
            raise ConnectionError("无法连接舵机驱动板，请检查I2C接线或地址。")

        h_servo = controller.setup_servo(
            H_SERVO_CHANNEL, max_angle=H_SERVO_MAX_ANGLE,
            min_safe_percent=H_SERVO_MIN_SAFE_PERCENT, max_safe_percent=H_SERVO_MAX_SAFE_PERCENT
        )
        v_servo = controller.setup_servo(
            V_SERVO_CHANNEL, max_angle=V_SERVO_MAX_ANGLE,
            min_safe_percent=V_SERVO_MIN_SAFE_PERCENT, max_safe_percent=V_SERVO_MAX_SAFE_PERCENT
        )

        gimbal = ServoGroup(h=h_servo, v=v_servo)
        
        logger.info("执行启动自检序列...")
        controller.run_initialization_sequence(duration_per_move=1.5)
        logger.info("舵机初始化完成。")

    except Exception as e:
        logger.critical(f"舵机系统初始化失败: {e}", exc_info=True)
        return

    # --- 初始化串口接收器 ---
    logger.info("[2/4] 初始化 MaixCam 串口接收器...")
    receiver = MaixCamReceiver(port=SERIAL_PORT, baudrate=BAUDRATE, debug=False)
    if not receiver.start():
        logger.critical("无法启动串口接收器，程序退出。")
        controller.cleanup()
        return
    logger.info(f"成功在后台启动接收线程，监听端口 {SERIAL_PORT}。")

    # --- 初始化PID控制器 ---
    logger.info("[3/4] 初始化 PID 控制器...")
    pid_h = PIDController(**base_pid_h, dead_zone=CENTER_DEAD_ZONE)
    pid_v = PIDController(**base_pid_v, dead_zone=CENTER_DEAD_ZONE)
    acquire_pid_h = PIDController(Kp=ACQUIRE_KP, Ki=0, Kd=0, setpoint=0.5)
    acquire_pid_v = PIDController(Kp=ACQUIRE_KP, Ki=0, Kd=0, setpoint=0.5)
    logger.info("PID控制器已创建。")

    # --- 初始化状态机和相关变量 ---
    logger.info("[4/4] 初始化状态机并进入主循环...")
    current_state = State.IDLE
    last_target_data = None
    last_data_timestamp = 0
    search_start_time = 0
    
    initial_pose = {'h': 50, 'v': 50}

    try:
        while True:
            target_data = receiver.get_latest_data()
            timestamp = time.time()

            # --- 状态转换逻辑 ---
            if target_data:
                last_target_data = target_data
                last_data_timestamp = timestamp
                if current_state == State.IDLE or current_state == State.SEARCHING:
                    current_state = State.ACQUIRING
                    logger.info(f"发现新目标，切换到状态: {current_state.name}")
                    acquire_pid_h.reset()
                    acquire_pid_v.reset()
            else:
                if current_state in [State.ACQUIRING, State.TRACKING] and timestamp - last_data_timestamp > TARGET_LOST_TIMEOUT:
                    current_state = State.SEARCHING
                    logger.warning(f"目标丢失，切换到状态: {current_state.name}")
                    search_start_time = timestamp
                elif current_state == State.SEARCHING and timestamp - search_start_time > SEARCH_TIMEOUT:
                    current_state = State.IDLE
                    logger.warning(f"搜索超时，切换到状态: {current_state.name}")
                    gimbal.move_to_pose(initial_pose, duration=2.0, easing_func=Easing.ease_in_out_cubic)

            # --- 状态行为处理 ---
            if current_state == State.ACQUIRING:
                norm_x = (last_target_data['x'] + last_target_data['w'] / 2) / CAM_WIDTH
                norm_y = (last_target_data['y'] + last_target_data['h'] / 2) / CAM_HEIGHT
                error = max(abs(norm_x - 0.5), abs(norm_y - 0.5))
                
                if error < ACQUIRE_THRESHOLD:
                    current_state = State.TRACKING
                    logger.info(f"目标捕获成功，切换到状态: {current_state.name}")
                    pid_h.reset()
                    pid_v.reset()
                else:
                    h_adj = acquire_pid_h.update(norm_x)
                    v_adj = acquire_pid_v.update(norm_y)
                    new_h = h_servo.current_position - h_adj
                    new_v = v_servo.current_position + v_adj
                    gimbal.move_to_pose({'h': new_h, 'v': new_v}, duration=0.1, easing_func=Easing.ease_in_out_cubic)
                    logger.debug(f"[ACQUIRING] H:{new_h:.1f} V:{new_v:.1f} Err:{error:.3f}")


            elif current_state == State.TRACKING:
                norm_x = (last_target_data['x'] + last_target_data['w'] / 2) / CAM_WIDTH
                norm_y = (last_target_data['y'] + last_target_data['h'] / 2) / CAM_HEIGHT
                norm_w = last_target_data['w'] / CAM_WIDTH

                if ADAPTIVE_GAIN_ENABLED and norm_w > 0.05:
                    gain_factor = 0.1 / norm_w 
                    pid_h.set_kp(base_pid_h['Kp'] * gain_factor)
                    pid_v.set_kp(base_pid_v['Kp'] * gain_factor)
                    logger.debug(f"[ADAPTIVE GAIN] Width:{norm_w:.2f}, Factor:{gain_factor:.2f}")

                h_adj = pid_h.update(norm_x)
                v_adj = pid_v.update(norm_y)
                
                new_h = h_servo.current_position - h_adj
                new_v = v_servo.current_position + v_adj
                gimbal.move_to_pose({'h': new_h, 'v': new_v}, duration=0.1, easing_func=Easing.ease_in_out_cubic)
                logger.debug(f"[TRACKING] H:{new_h:.1f} V:{new_v:.1f} H_adj:{h_adj:.2f} V_adj:{v_adj:.2f}")

            elif current_state == State.SEARCHING:
                elapsed_search_time = timestamp - search_start_time
                t = elapsed_search_time / SEARCH_TIMEOUT
                
                last_norm_x = (last_target_data['x'] + last_target_data['w'] / 2) / CAM_WIDTH - 0.5
                last_norm_y = (last_target_data['y'] + last_target_data['h'] / 2) / CAM_HEIGHT - 0.5
                
                offset_x, offset_y = get_spiral_point(t, SEARCH_SPIRAL_MAX_RADIUS, SEARCH_SPIRAL_DENSITY)
                
                target_x = 0.5 + last_norm_x + offset_x
                target_y = 0.5 + last_norm_y + offset_y
                
                new_h = target_x * 100
                new_v = target_y * 100
                gimbal.move_to_pose({'h': new_h, 'v': new_v}, duration=0.2, easing_func=Easing.linear)
                logger.debug(f"[SEARCHING] T:{t:.2f} H:{new_h:.1f} V:{new_v:.1f}")
            
            # 主循环延时
            time.sleep(0.02)

    except KeyboardInterrupt:
        logger.info("程序被用户中断。")
    except Exception as e:
        logger.critical("主循环发生未捕获的异常", exc_info=True)
    finally:
        # --- 清理资源 ---
        logger.info("正在执行清理操作...")
        receiver.stop()
        
        logger.info("云台最终归中...")
        gimbal.move_to_pose(initial_pose, duration=1.5).wait_for_move()
        
        controller.cleanup()
        logger.info("清理完成。程序安全退出。")

if __name__ == "__main__":
    main()
