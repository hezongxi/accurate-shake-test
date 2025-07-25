#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 立即显示启动信息（在任何导入之前）
import sys
print("", flush=True)  # 强制刷新输出缓冲区
print("="*60, flush=True)
print("防抖振动测试程序 v1.0", flush=True)
print("="*60, flush=True)
print("程序启动中...", flush=True)
print("正在初始化，预计需要10-30秒，请耐心等待...", flush=True)
print("", flush=True)

"""
防抖振动测试程序
不使用卡尔曼滤波，确保测量真实的抖动数据
只在检测阶段使用稳定算法，在测量阶段记录原始数据
"""

print("正在导入核心库文件...", flush=True)
import cv2
print("✓ OpenCV库导入完成", flush=True)
import numpy as np
print("✓ NumPy库导入完成", flush=True)
import os
from collections import deque
import math
import threading
import time
import datetime
try:
    import ctypes
    from ctypes import wintypes
    WINDOWS_API_AVAILABLE = True
    print("✓ Windows API导入完成", flush=True)
except ImportError:
    WINDOWS_API_AVAILABLE = False
    print("ⓘ Windows API不可用，使用基础键盘输入", flush=True)
print("✓ 所有库导入完成!", flush=True)
print("✓ 程序初始化完成，正在启动主界面...", flush=True)
print("")

class AccurateShakeTest:
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.running = False
        self.cap = None
        self.frame_count = 0
        
        # 全局键盘状态
        self.global_key_pressed = None
        self.keyboard_thread = None
        self.keyboard_running = False
        
        # 测试结果记录
        self.test_results = []
        
        # 分离检测和测量
        self.raw_positions = []  # 原始检测位置（用于抖动测量）
        self.stable_positions = []  # 稳定位置（用于显示）
        self.reference_center = None
        
        # 检测稳定性参数
        self.detection_history = deque(maxlen=5)
        self.roi_size = 100
        self.detection_confidence = []
        
        # 实时极值点追踪
        self.max_x_pos = None
        self.min_x_pos = None
        self.max_y_pos = None
        self.min_y_pos = None
        self.max_x_val = None
        self.min_x_val = None
        self.max_y_val = None
        self.min_y_val = None
        
        # 用户选择的ROI区域
        self.user_roi = None
        self.selecting_roi = False
        self.roi_start_point = None
        self.roi_end_point = None
        
        
        # 用于显示的模板匹配图像
        self.template_match_image = None
        self.template_image = None
        
        # 按键防抖
        self.last_key_time = 0
        self.key_debounce_delay = 0.3  # 300ms防抖延迟
        
        
        # 自动切换到英文输入法
        self.switch_to_english_keyboard()
    
    def update_template_match_image_for_display(self, roi, detected_position=None):
        """更新用于显示的模板匹配图像"""
        # 检查图像是否已经是灰度图
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi.copy()
        
        # 对模糊图像进行预处理（与模板匹配相同的处理）
        blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=10)
        
        # 转换为三通道以便画彩色标记
        template_display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # 如果识别到了十字，在图像上画出十字模板的位置
        if detected_position is not None and self.user_roi is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = self.user_roi
            rx, ry = detected_position
            
            # 检查检测点是否在ROI内
            if roi_x1 <= rx < roi_x2 and roi_y1 <= ry < roi_y2:
                # 转换为ROI内的相对坐标
                roi_x = rx - roi_x1
                roi_y = ry - roi_y1
                
                # 画出模板匹配找到的十字位置
                template_size = 24
                half_size = template_size // 2
                
                # 画十字线（蓝色）
                cv2.line(template_display, 
                        (max(0, roi_x - half_size + 2), roi_y), 
                        (min(enhanced.shape[1], roi_x + half_size - 2), roi_y), 
                        (255, 0, 0), 2)
                cv2.line(template_display, 
                        (roi_x, max(0, roi_y - half_size + 2)), 
                        (roi_x, min(enhanced.shape[0], roi_y + half_size - 2)), 
                        (255, 0, 0), 2)
                
                # 画中心点（红色）
                cv2.circle(template_display, (roi_x, roi_y), 3, (0, 0, 255), -1)
        
        # 保存处理后的图像用于显示
        self.template_match_image = template_display
    
    def switch_to_english_keyboard(self):
        """切换键盘到英文模式"""
        try:
            if WINDOWS_API_AVAILABLE:
                import ctypes
                from ctypes import wintypes
                
                # 定义常量
                HKL_NEXT = 1
                HKL_PREV = 0
                
                user32 = ctypes.windll.user32
                
                # 获取所有已安装的键盘布局
                layouts = (wintypes.HKL * 10)()
                count = user32.GetKeyboardLayoutList(10, layouts)
                
                # 查找英文布局 (0x0409 = 美式英语)
                english_hkl = None
                for i in range(count):
                    layout_id = layouts[i] & 0xFFFF
                    if layout_id == 0x0409:  # 美式英语
                        english_hkl = layouts[i]
                        break
                
                if english_hkl:
                    # 切换到英文布局
                    hwnd = user32.GetForegroundWindow()
                    result = user32.ActivateKeyboardLayout(english_hkl, 0)
                    if result:
                        print("已切换到英文输入模式")
                        return
                
                # 如果找不到英文布局，尝试使用Ctrl+Shift切换
                print("使用快捷键切换输入法")
                import time
                user32.keybd_event(0x11, 0, 0, 0)  # Ctrl down
                user32.keybd_event(0x10, 0, 0, 0)  # Shift down
                time.sleep(0.1)
                user32.keybd_event(0x10, 0, 2, 0)  # Shift up
                user32.keybd_event(0x11, 0, 2, 0)  # Ctrl up
                print("已尝试使用Ctrl+Shift切换输入法")
                
            else:
                print("Windows API不可用，跳过输入法切换")
        except Exception as e:
            print(f"切换输入法失败: {e}")
            print("请手动切换到英文输入模式以确保按键响应正常")
    
    def start_global_keyboard_listener(self):
        """启动全局键盘监听"""
        if not WINDOWS_API_AVAILABLE:
            return
            
        def keyboard_listener():
            self.keyboard_running = True
            user32 = ctypes.windll.user32
            
            # 定义按键码
            VK_P = 0x50
            VK_Q = 0x51  
            VK_R = 0x52
            VK_ESCAPE = 0x1B
            VK_SPACE = 0x20
            VK_OEM_PLUS = 0xBB  # +号键
            VK_OEM_MINUS = 0xBD  # -号键
            
            last_pressed = {}
            
            while self.keyboard_running:
                try:
                    current_time = time.time()
                    
                    # 检查各个按键状态
                    if user32.GetAsyncKeyState(VK_P) & 0x8000:
                        if VK_P not in last_pressed or current_time - last_pressed[VK_P] > 0.3:
                            self.global_key_pressed = ord('p')
                            last_pressed[VK_P] = current_time
                    elif user32.GetAsyncKeyState(VK_Q) & 0x8000:
                        if VK_Q not in last_pressed or current_time - last_pressed[VK_Q] > 0.3:
                            self.global_key_pressed = ord('q')
                            last_pressed[VK_Q] = current_time
                    elif user32.GetAsyncKeyState(VK_R) & 0x8000:
                        if VK_R not in last_pressed or current_time - last_pressed[VK_R] > 0.3:
                            self.global_key_pressed = ord('r')
                            last_pressed[VK_R] = current_time
                    elif user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                        if VK_ESCAPE not in last_pressed or current_time - last_pressed[VK_ESCAPE] > 0.3:
                            self.global_key_pressed = 27
                            last_pressed[VK_ESCAPE] = current_time
                    elif user32.GetAsyncKeyState(VK_SPACE) & 0x8000:
                        if VK_SPACE not in last_pressed or current_time - last_pressed[VK_SPACE] > 0.3:
                            self.global_key_pressed = ord(' ')
                            last_pressed[VK_SPACE] = current_time
                    elif user32.GetAsyncKeyState(VK_OEM_PLUS) & 0x8000:
                        if VK_OEM_PLUS not in last_pressed or current_time - last_pressed[VK_OEM_PLUS] > 0.3:
                            self.global_key_pressed = ord('+')
                            last_pressed[VK_OEM_PLUS] = current_time
                    elif user32.GetAsyncKeyState(VK_OEM_MINUS) & 0x8000:
                        if VK_OEM_MINUS not in last_pressed or current_time - last_pressed[VK_OEM_MINUS] > 0.3:
                            self.global_key_pressed = ord('-')
                            last_pressed[VK_OEM_MINUS] = current_time
                    
                    time.sleep(0.05)  # 减少CPU使用
                except Exception:
                    break
                    
        if WINDOWS_API_AVAILABLE:
            self.keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
            self.keyboard_thread.start()
    
    def stop_global_keyboard_listener(self):
        """停止全局键盘监听"""
        self.keyboard_running = False
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=1)
    
    
    
    
    
    def get_key_input(self, timeout=1):
        """获取键盘输入，优先使用全局监听"""
        # 先检查全局键盘监听
        if WINDOWS_API_AVAILABLE and self.global_key_pressed is not None:
            key = self.global_key_pressed
            self.global_key_pressed = None  # 清除状态
            return key
        
        # 回退到OpenCV的waitKey，确保能正确检测按键
        key = cv2.waitKey(timeout) & 0xFF
        if key == 255:  # 没有按键时返回255
            return -1
        return key
    
    def save_test_result(self, video_name, y_range, max_y_val, min_y_val, x_range, max_x_val, min_x_val, max_shake, frame_count):
        """保存单个视频的测试结果"""
        
        test_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_result = f"通过" if max_shake <= 10 else "失败"
        
        result_data = {
            'video_name': video_name,
            'test_time': test_time,
            'frame_count': frame_count,
            'y_range': y_range,
            'max_y_val': max_y_val,
            'min_y_val': min_y_val,
            'x_range': x_range,
            'max_x_val': max_x_val,
            'min_x_val': min_x_val,
            'max_shake': max_shake,
            'result': test_result
        }
        
        self.test_results.append(result_data)
        self.export_results_to_file()
    
    def export_results_to_file(self):
        """导出所有测试结果到txt文件"""
        try:
            with open('测试报告.txt', 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("测试结果报告\n")
                f.write("="*80 + "\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"测试标准: 振动幅度 ≤ 10像素为合格\n")
                f.write(f"共测试视频: {len(self.test_results)} 个\n")
                f.write("="*80 + "\n\n")
                
                for i, result in enumerate(self.test_results, 1):
                    f.write(f"【测试 {i}】\n")
                    f.write(f"视频文件: {result['video_name']}\n")
                    f.write(f"测试时间: {result['test_time']}\n")
                    f.write(f"总帧数: {result['frame_count']}\n")
                    f.write(f"\n测试结果:\n")
                    
                    if result['max_y_val'] is not None:
                        f.write(f"Y方向: 最低点={result['min_y_val']:.0f}, 最高点={result['max_y_val']:.0f}\n")
                        f.write(f"Y方向像素点差量: {result['y_range']:.1f} pixel ({result['max_y_val']:.0f} - {result['min_y_val']:.0f})\n")
                    else:
                        f.write(f"Y方向像素点差量: {result['y_range']:.1f} pixel\n")
                    
                    if result['max_x_val'] is not None:
                        f.write(f"X方向: 最左点={result['min_x_val']:.0f}, 最右点={result['max_x_val']:.0f}\n")
                        f.write(f"X方向像素点差量: {result['x_range']:.1f} pixel ({result['max_x_val']:.0f} - {result['min_x_val']:.0f})\n")
                    else:
                        f.write(f"X方向像素点差量: {result['x_range']:.1f} pixel\n")
                    
                    f.write(f"最大振动值: {result['max_shake']:.1f} pixel (取X、Y方向的最大值)\n")
                    f.write(f"防抖振动测试结果: {result['result']}\n")
                    if result['max_shake'] > 10:
                        f.write(f"超标: {result['max_shake'] - 10:.1f} pixel\n")
                    f.write("\n" + "-"*60 + "\n\n")
                
                # 统计摘要
                passed_count = sum(1 for r in self.test_results if r['result'] == '通过')
                failed_count = len(self.test_results) - passed_count
                
                f.write("="*80 + "\n")
                f.write("测试统计摘要\n")
                f.write("="*80 + "\n")
                f.write(f"总测试数: {len(self.test_results)}\n")
                f.write(f"通过数量: {passed_count}\n")
                f.write(f"失败数量: {failed_count}\n")
                f.write(f"通过率: {passed_count/len(self.test_results)*100:.1f}%\n")
                
                if failed_count > 0:
                    f.write(f"\n失败的视频:\n")
                    for result in self.test_results:
                        if result['result'] == '失败':
                            f.write(f"- {result['video_name']} (振动值: {result['max_shake']:.1f} pixel)\n")
                
            print(f"✓ 测试结果已保存到: 测试报告.txt")
            
        except Exception as e:
            print(f"保存测试结果失败: {e}")
    
    def initialize_results_file(self):
        """初始化结果文件（程序启动时清空）"""
        self.test_results = []
        try:
            # 创建空的结果文件
            with open('测试报告.txt', 'w', encoding='utf-8') as f:
                f.write("测试结果报告\n")
                f.write("等待测试结果...\n")
        except Exception as e:
            print(f"初始化结果文件失败: {e}")
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于选择ROI区域"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting_roi = True
            self.roi_start_point = (x, y)
            self.roi_end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting_roi:
                self.roi_end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selecting_roi:
                self.selecting_roi = False
                self.roi_end_point = (x, y)
                
                # 确保ROI区域有效
                if self.roi_start_point and self.roi_end_point:
                    x1, y1 = self.roi_start_point
                    x2, y2 = self.roi_end_point
                    
                    # 确保坐标顺序正确
                    roi_x1 = min(x1, x2)
                    roi_y1 = min(y1, y2)
                    roi_x2 = max(x1, x2)
                    roi_y2 = max(y1, y2)
                    
                    # 检查ROI大小
                    roi_width = roi_x2 - roi_x1
                    roi_height = roi_y2 - roi_y1
                    print(f"ROI尺寸: {roi_width}x{roi_height}")
                    if roi_width > 30 and roi_height > 30:
                        self.user_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
                        print(f"✓ ROI区域已选择: ({roi_x1}, {roi_y1}) 到 ({roi_x2}, {roi_y2})")
                        print("准备自动开始分析...")
                        # 设置自动开始标记
                        if hasattr(self, 'roi_auto_start_flag'):
                            self.roi_auto_start_flag = True
                    else:
                        print("✗ ROI区域太小，请重新选择（最小30x30像素）")
                        
    def get_roi_for_detection(self, frame, search_center):
        """获取用于检测的ROI区域"""
        if self.user_roi:
            # 使用用户选择的ROI
            roi_x1, roi_y1, roi_x2, roi_y2 = self.user_roi
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            return roi, (roi_x1, roi_y1)
        else:
            # 使用原来的动态ROI
            return self.get_roi(frame, search_center)
        
    def detect_cross_multiple_methods(self, frame, search_center):
        """使用多种方法检测十字交叉点，返回所有候选点和置信度"""
        candidates = []
        
        # 获取ROI
        roi, roi_offset = self.get_roi_for_detection(frame, search_center)
        if roi.size == 0:
            return candidates
            
        # 只使用模板匹配检测
        template_result = self.detect_by_template(roi, roi_offset)
        if template_result:
            candidates.append(template_result)
            print(f"  - template方法检测成功，置信度: {template_result['confidence']:.3f}")
        else:
            print(f"  - template方法检测失败")
            
        return candidates
    
    def get_roi(self, frame, center):
        """获取检测ROI区域"""
        height, width = frame.shape[:2]
        x, y = center
        
        x1 = max(0, x - self.roi_size // 2)
        y1 = max(0, y - self.roi_size // 2)
        x2 = min(width, x + self.roi_size // 2)
        y2 = min(height, y + self.roi_size // 2)
        
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1)
    
    def detect_by_template(self, roi, roi_offset):
        """模板匹配检测"""
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 对模糊图像进行预处理
        # 应用高斯模糊减少噪声，然后增强对比度
        blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=10)
        
        # 创建十字模板
        template_size = 24
        template = np.ones((template_size, template_size), dtype=np.uint8) * 255
        center = template_size // 2
        
        # 画十字
        cv2.line(template, (2, center), (template_size-3, center), 0, 2)
        cv2.line(template, (center, 2), (center, template_size-3), 0, 2)
        # 中心点
        cv2.circle(template, (center, center), 2, 0, -1)
        
        # 模板匹配 - 对增强后的图像
        result = cv2.matchTemplate(enhanced, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 根据图像质量动态调整阈值
        roi_contrast = gray_roi.std()
        if roi_contrast < 30:  # 低对比度图像
            threshold = 0.4  # 降低阈值
        else:
            threshold = 0.6  # 正常阈值
        
        if max_val > threshold:
            roi_x = max_loc[0] + template_size // 2
            roi_y = max_loc[1] + template_size // 2
            global_pos = (roi_x + roi_offset[0], roi_y + roi_offset[1])
            return {'position': global_pos, 'confidence': max_val, 'method': 'template'}
        
        return None
    
    
    
    def select_best_detection(self, candidates, search_center):
        """从候选点中选择最佳检测结果"""
        if not candidates:
            return None
        
        # 现在只有模板匹配一种方法，直接返回第一个结果
        return candidates[0]
    
    


    def detect_cross_center(self, frame):
        """主检测函数"""
        # 确定搜索中心
        if len(self.stable_positions) > 0:
            search_center = self.stable_positions[-1]
        elif self.reference_center:
            search_center = self.reference_center
        else:
            search_center = (frame.shape[1]//2, frame.shape[0]//2)
        
        # 先获取ROI用于检测
        roi, roi_offset = self.get_roi_for_detection(frame, search_center)
        
        # 检测候选点
        candidates = self.detect_cross_multiple_methods(frame, search_center)
        
        # 选择最佳检测结果
        best_detection = self.select_best_detection(candidates, search_center)
        
        if best_detection:
            raw_position = best_detection['position']
            confidence = best_detection['confidence']
            method = best_detection['method']
            
            # 检查置信度阈值
            if confidence < 0.7:
                print(f"帧 {self.frame_count}: {method} 方法置信度过低 ({confidence:.3f} < 0.7)，跳过该帧")
                
                # 更新模板匹配显示图像，不显示十字（置信度过低）
                self.update_template_match_image_for_display(roi, None)
                
                # 使用上一个位置
                if self.raw_positions:
                    last_raw = self.raw_positions[-1]
                    last_stable = self.stable_positions[-1]
                    self.raw_positions.append(last_raw)
                    self.stable_positions.append(last_stable)
                    return last_raw, last_stable, 0.0
                else:
                    # 如果没有历史位置，使用搜索中心
                    self.raw_positions.append(search_center)
                    self.stable_positions.append(search_center)
                    return search_center, search_center, 0.0
            
            # 打印检测方法信息
            print(f"帧 {self.frame_count}: 使用 {method} 方法检测到十字, 置信度: {confidence:.3f}")
            
            # 记录原始位置（用于抖动测量）
            self.raw_positions.append(raw_position)
            self.detection_confidence.append(confidence)
            
            # 设置参考中心（如果还没有设置）
            reference_center_just_set = False
            if self.reference_center is None:
                self.reference_center = raw_position
                reference_center_just_set = True
                print(f"参考中心设置为: {raw_position}")
                # 重置极值点，避免参考中心被算入抖动范围
                self.reset_extreme_points()
            
            # 实时更新极值点（参考中心不会被重复添加）
            self.update_extreme_points(raw_position)
            
            # 添加到历史记录
            self.detection_history.append(raw_position)
            
            # 计算稳定位置（仅用于显示，不影响测量）
            if len(self.detection_history) >= 3:
                # 使用最近3个位置的中位数作为稳定位置
                recent_positions = list(self.detection_history)[-3:]
                x_coords = [pos[0] for pos in recent_positions]
                y_coords = [pos[1] for pos in recent_positions]
                
                stable_x = int(np.median(x_coords))
                stable_y = int(np.median(y_coords))
                stable_position = (stable_x, stable_y)
            else:
                stable_position = raw_position
            
            self.stable_positions.append(stable_position)
            
            # 更新模板匹配显示图像，显示检测到的十字
            self.update_template_match_image_for_display(roi, raw_position)
            
            return raw_position, stable_position, confidence
        
        # 如果检测失败，使用上一个位置
        if self.raw_positions:
            last_raw = self.raw_positions[-1]
            last_stable = self.stable_positions[-1]
            self.raw_positions.append(last_raw)
            self.stable_positions.append(last_stable)
            
            # 更新模板匹配显示图像，不显示十字（检测失败）
            self.update_template_match_image_for_display(roi, None)
            
            return last_raw, last_stable, 0.0
        
        # 完全失败，使用搜索中心
        self.raw_positions.append(search_center)
        self.stable_positions.append(search_center)
        
        # 更新模板匹配显示图像，不显示十字（完全失败）
        self.update_template_match_image_for_display(roi, None)
        
        return search_center, search_center, 0.0
    
    def update_extreme_points(self, new_position):
        """实时更新极值点"""
        x, y = new_position
        
        # 更新X方向极值
        if self.max_x_val is None or x > self.max_x_val:
            self.max_x_val = x
            self.max_x_pos = new_position
            
        if self.min_x_val is None or x < self.min_x_val:
            self.min_x_val = x
            self.min_x_pos = new_position
            
        # 更新Y方向极值
        if self.max_y_val is None or y > self.max_y_val:
            self.max_y_val = y
            self.max_y_pos = new_position
            
        if self.min_y_val is None or y < self.min_y_val:
            self.min_y_val = y
            self.min_y_pos = new_position
    
    def reset_extreme_points(self):
        """重置极值点"""
        self.max_x_pos = None
        self.min_x_pos = None
        self.max_y_pos = None
        self.min_y_pos = None
        self.max_x_val = None
        self.min_x_val = None
        self.max_y_val = None
        self.min_y_val = None
    
    def calculate_shake_metrics(self):
        """计算振动指标 - 按照防抖振动测试标准：找出最高点与最低点，然后计算差量"""
        if len(self.raw_positions) < 2:
            return 0, None, None, 0, None, None
        
        # 使用实时更新的极值点（ROI窗口显示的数据）
        if (self.max_y_val is not None and self.min_y_val is not None and 
            self.max_x_val is not None and self.min_x_val is not None):
            
            # 计算Y方向差量
            y_range = self.max_y_val - self.min_y_val
            
            # 计算X方向差量
            x_range = self.max_x_val - self.min_x_val
            
            return y_range, self.max_y_val, self.min_y_val, x_range, self.max_x_val, self.min_x_val
        else:
            # 如果极值点还没有初始化，返回0值
            return 0, None, None, 0, None, None
    
    def draw_results(self, frame, raw_pos, stable_pos, confidence):
        """绘制检测结果"""
        height, width = frame.shape[:2]
        
        # 绘制ROI框
        if self.user_roi:
            # 绘制用户选择的固定ROI
            roi_x1, roi_y1, roi_x2, roi_y2 = self.user_roi
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
        elif len(self.stable_positions) > 0:
            # 绘制动态ROI
            search_center = self.stable_positions[-1]
            roi_x1 = max(0, search_center[0] - self.roi_size // 2)
            roi_y1 = max(0, search_center[1] - self.roi_size // 2)
            roi_x2 = min(width, search_center[0] + self.roi_size // 2)
            roi_y2 = min(height, search_center[1] + self.roi_size // 2)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 1)
        # 绘制正在选择的ROI
        if self.selecting_roi and self.roi_start_point and self.roi_end_point:
            cv2.rectangle(frame, self.roi_start_point, self.roi_end_point, (0, 255, 0), 2)
        
        # 绘制十字线（在原始位置）
        cross_length = 35
        raw_x, raw_y = raw_pos
        cv2.line(frame, 
                (max(0, raw_x - cross_length), raw_y), 
                (min(width, raw_x + cross_length), raw_y), 
                (0, 255, 0), 2)
        
        cv2.line(frame, 
                (raw_x, max(0, raw_y - cross_length)), 
                (raw_x, min(height, raw_y + cross_length)), 
                (0, 255, 0), 2)
        
        # 绘制当前检测点（红色）- 用于测量
        cv2.circle(frame, raw_pos, 3, (0, 0, 255), -1)
        cv2.circle(frame, raw_pos, 6, (0, 0, 255), 2)
        
        # 绘制稳定位置（橙色）- 用于显示
        if raw_pos != stable_pos:
            cv2.circle(frame, stable_pos, 2, (0, 165, 255), -1)
        
        # 绘制Y方向极值点
        # Y方向最高点（粉色）
        if self.max_y_pos:
            cv2.circle(frame, self.max_y_pos, 4, (255, 192, 203), -1)
            cv2.circle(frame, self.max_y_pos, 8, (255, 192, 203), 2)
            cv2.putText(frame, f"Y-MAX({self.max_y_val})", 
                       (self.max_y_pos[0] - 80, self.max_y_pos[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 192, 203), 1)
        
        # Y方向最低点（深绿色）
        if self.min_y_pos:
            cv2.circle(frame, self.min_y_pos, 4, (0, 128, 0), -1)
            cv2.circle(frame, self.min_y_pos, 8, (0, 128, 0), 2)
            cv2.putText(frame, f"Y-MIN({self.min_y_val})", 
                       (self.min_y_pos[0] - 80, self.min_y_pos[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 0), 1)
        
        # 绘制X方向极值点
        # X方向最右点（紫色）
        if self.max_x_pos:
            cv2.circle(frame, self.max_x_pos, 4, (128, 0, 128), -1)
            cv2.circle(frame, self.max_x_pos, 8, (128, 0, 128), 2)
            cv2.putText(frame, f"X-MAX({self.max_x_val})", 
                       (self.max_x_pos[0] + 10, self.max_x_pos[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)
        
        # X方向最左点（橙色）
        if self.min_x_pos:
            cv2.circle(frame, self.min_x_pos, 4, (0, 165, 255), -1)
            cv2.circle(frame, self.min_x_pos, 8, (0, 165, 255), 2)
            cv2.putText(frame, f"X-MIN({self.min_x_val})", 
                       (self.min_x_pos[0] + 10, self.min_x_pos[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        # 绘制原始位置轨迹
        if len(self.raw_positions) > 1:
            points = self.raw_positions[-20:]  # 最近20个点
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], (128, 128, 128), 1)
        
        return frame
    
    def run_test(self):
        """运行测试"""
        if not self.video_path or not os.path.exists(self.video_path):
            print("错误：视频文件不存在")
            return

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print("错误：无法打开视频文件")
            return

        # 获取视频信息
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息:")
        print(f"分辨率: {width}x{height}")
        print(f"帧率: {fps:.2f} FPS")
        print(f"总帧数: {total_frames}")
        print(f"时长: {total_frames/fps:.2f} 秒")
        print()

        # 计算窗口大小（屏幕高的80%，宽高比不变）
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_height = user32.GetSystemMetrics(1)
        except Exception:
            screen_height = 1080  # 默认值
        rotated_width = height  # 旋转后宽高互换
        rotated_height = width
        aspect_ratio = rotated_width / rotated_height
        window_height = int(screen_height * 0.8)
        window_width = int(window_height * aspect_ratio)

        # 创建窗口并设置鼠标回调
        window_name = 'Anti-Shake Test Program'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        # 禁止窗口缩放（Win32 API，移除最大化和拉伸）
        try:
            import ctypes
            # 使用英文窗口名避免编码问题
            hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
            if hwnd:  # 只有找到窗口时才修改样式
                GWL_STYLE = -16
                WS_THICKFRAME = 0x00040000
                WS_MAXIMIZEBOX = 0x00010000
                style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
                style &= ~WS_THICKFRAME
                style &= ~WS_MAXIMIZEBOX
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)
        except Exception as e:
            print(f"窗口样式设置失败: {e}")  # 输出错误但不影响程序运行

        self.running = True
        self.frame_count = 0
        
        # 启动全局键盘监听
        print("启动全局键盘监听...")
        self.start_global_keyboard_listener()

        # 播放控制参数
        paused = True
        step_mode = False
        play_speed = 3.0  # 默认3倍速播放
        frame_delay = max(1, int(1000 / fps / play_speed))

        # --- 初始帧与ROI选择 ---
        print("=== ROI 选择阶段 ===")
        print("1. 在图像上拖动鼠标来选择一个区域.")
        print("2. 选择完成后会自动开始播放和分析.")
        print("3. 按 'q' 或 ESC 键退出.")
        
        ret, first_frame = self.cap.read()
        if not ret:
            print("错误：无法读取视频的第一帧")
            self.cap.release()
            cv2.destroyAllWindows()
            return
        # 旋转第一帧
        first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
        self.current_frame = first_frame.copy()  # 保存当前帧

        # 初始化自动开始标记
        self.roi_auto_start_flag = False
        
        while self.running:
            frame_for_roi = first_frame.copy()
            # 绘制正在选择的ROI
            if self.selecting_roi and self.roi_start_point and self.roi_end_point:
                cv2.rectangle(frame_for_roi, self.roi_start_point, self.roi_end_point, (0, 255, 0), 2)
            # 绘制已选择的ROI
            elif self.user_roi:
                 roi_x1, roi_y1, roi_x2, roi_y2 = self.user_roi
                 cv2.rectangle(frame_for_roi, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

            cv2.imshow(window_name, frame_for_roi)
            
            # 如果ROI已选择且设置了自动开始标记，则直接开始播放
            if self.user_roi and self.roi_auto_start_flag:
                print(f"ROI确认: {self.user_roi}. 自动开始分析...")
                break
            
            key = self.get_key_input(30)

            if key == ord('q') or key == 27:
                self.running = False
                break

        if not self.running:
            self.cap.release()
            cv2.destroyAllWindows()
            return
            
        # ROI选择阶段结束，禁用鼠标框选功能
        self.roi_selection_enabled = False
        
        # 在选择的ROI内检测第一个交叉点
        self.frame_count += 1
        raw_pos, stable_pos, confidence = self.detect_cross_center(first_frame)

        # --- 主播放与分析循环 ---
        print("\n=== 分析阶段 ===")
        print("控制说明:")
        print("  按 'q' 或 ESC 退出")
        print("  按 'p' 暂停/继续")
        print("  按 '+' 加速，'-' 减速")
        print("  按 'space' 单帧步进 (暂停时)")
        print("开始自动播放...")
        

        paused = False # 开始自动播放
        
        while self.running:
            # 播放控制逻辑
            if not paused and not step_mode:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("视频播放完毕")
                    break
                    
                # 旋转帧
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                self.current_frame = frame.copy()  # 保存当前帧
                self.frame_count += 1
                
                # 检测十字中心
                raw_pos, stable_pos, confidence = self.detect_cross_center(frame)
            
            # 单帧步进模式
            elif step_mode:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("视频播放完毕")
                    break
                    
                # 旋转帧
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                self.current_frame = frame.copy()  # 保存当前帧
                self.frame_count += 1
                
                # 检测十字中心
                raw_pos, stable_pos, confidence = self.detect_cross_center(frame)
                
                step_mode = False  # 单帧步进后停止
            
            
            # 显示当前帧（如果有）
            if 'frame' in locals():
                # 绘制结果
                display_frame = self.draw_results(frame.copy(), raw_pos, stable_pos, confidence)
                # 计算抖动指标
                y_range, max_y_val, min_y_val, x_range, max_x_val, min_x_val = self.calculate_shake_metrics()
                max_shake = max(y_range, x_range)  # 取X和Y方向的最大抖动值
                # 删除左上角所有文字
                cv2.imshow(window_name, display_frame)
                # ROI放大预览
                preview_window = 'ROI Preview'
                show_preview = False
                preview_img = None
                if self.user_roi:
                    roi_x1, roi_y1, roi_x2, roi_y2 = self.user_roi
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi.size > 0:
                        preview_img = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                        # 画交叉点（如果在ROI内）
                        if self.raw_positions:
                            last_raw = self.raw_positions[-1]
                            rx, ry = last_raw
                            if roi_x1 <= rx < roi_x2 and roi_y1 <= ry < roi_y2:
                                px = int((rx - roi_x1) * 2)
                                py = int((ry - roi_y1) * 2)
                                cv2.drawMarker(preview_img, (px, py), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
                        
                        # 在左上角显示帧数和坐标信息
                        line_height = 25
                        y_pos = 20
                        
                        # 帧数
                        frame_text = f"Frame: {self.frame_count}"
                        cv2.putText(preview_img, frame_text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        y_pos += line_height
                        
                        # 当前识别点坐标
                        if self.raw_positions:
                            current_pos = self.raw_positions[-1]
                            current_text = f"Current: ({current_pos[0]}, {current_pos[1]})"
                            cv2.putText(preview_img, current_text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            y_pos += line_height
                        
                        # Y方向极值点和差值
                        if self.max_y_val is not None and self.min_y_val is not None:
                            y_max_text = f"Y-Max: ({self.max_y_pos[0] if self.max_y_pos else 0}, {self.max_y_val})"
                            y_min_text = f"Y-Min: ({self.min_y_pos[0] if self.min_y_pos else 0}, {self.min_y_val})"
                            y_diff_text = f"Y-Diff: {self.max_y_val - self.min_y_val:.1f}"
                            
                            cv2.putText(preview_img, y_max_text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 139), 2)
                            y_pos += line_height
                            cv2.putText(preview_img, y_min_text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 0), 2)
                            y_pos += line_height
                            cv2.putText(preview_img, y_diff_text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
                            y_pos += line_height
                        
                        # X方向极值点和差值
                        if self.max_x_val is not None and self.min_x_val is not None:
                            x_max_text = f"X-Max: ({self.max_x_val}, {self.max_x_pos[1] if self.max_x_pos else 0})"
                            x_min_text = f"X-Min: ({self.min_x_val}, {self.min_x_pos[1] if self.min_x_pos else 0})"
                            x_diff_text = f"X-Diff: {self.max_x_val - self.min_x_val:.1f}"
                            
                            cv2.putText(preview_img, x_max_text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 2)
                            y_pos += line_height
                            cv2.putText(preview_img, x_min_text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)
                            y_pos += line_height
                            cv2.putText(preview_img, x_diff_text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
                        
                        show_preview = True
                        
                # 显示模板匹配图像窗口
                template_window = 'Template Match ROI'
                if self.template_match_image is not None:
                    # 放大模板匹配图像以便更好观察
                    template_resized = cv2.resize(self.template_match_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                    # template_match_image 已经是三通道BGR图像了，直接使用
                    template_display = template_resized
                    
                    # 不再在模板匹配图像上额外标记交叉点，因为已经在update_template_match_image_for_display中处理了
                    
                    cv2.imshow(template_window, template_display)
                else:
                    # 没有模板匹配图像时显示占位图
                    if self.user_roi:
                        roi_x1, roi_y1, roi_x2, roi_y2 = self.user_roi
                        roi_width = roi_x2 - roi_x1
                        roi_height = roi_y2 - roi_y1
                        placeholder = np.zeros((roi_height * 2, roi_width * 2, 3), dtype=np.uint8)
                        cv2.putText(placeholder, 'No Template Data', (10, roi_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                        cv2.imshow(template_window, placeholder)
                
                # 显示十字模板窗口
                template_pattern_window = 'Cross Template'
                if self.template_image is not None:
                    # 放大模板以便更好观察
                    template_pattern_resized = cv2.resize(self.template_image, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(template_pattern_window, template_pattern_resized)
                        
                if show_preview and preview_img is not None:
                    cv2.imshow(preview_window, preview_img)
                else:
                    try:
                        cv2.destroyWindow(preview_window)
                    except cv2.error:
                        pass  # 忽略窗口不存在的错误
            
            # 处理按键 - 暂停时使用较长延迟以减少CPU占用
            key_delay = 50 if paused else frame_delay
            key = self.get_key_input(key_delay)
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                current_time = time.time()
                if current_time - self.last_key_time > self.key_debounce_delay:
                    paused = not paused
                    print(f"{'暂停' if paused else '继续'}播放")
                    self.last_key_time = current_time
            elif key == ord('+') or key == ord('='):
                play_speed = min(5.0, play_speed + 0.2)
                frame_delay = max(1, int(1000 / fps / play_speed))
                print(f"播放速度: {play_speed:.1f}x")
            elif key == ord('-') or key == ord('_'):
                play_speed = max(0.1, play_speed - 0.2)
                frame_delay = max(1, int(1000 / fps / play_speed))
                print(f"播放速度: {play_speed:.1f}x")
            elif key == ord(' '):  # 空格键单帧步进
                current_time = time.time()
                if paused and current_time - self.last_key_time > self.key_debounce_delay:
                    step_mode = True
                    self.last_key_time = current_time
                
        # 清理资源
        self.stop_global_keyboard_listener()
        self.cap.release()
        cv2.destroyAllWindows()
        
        # 最终结果
        if self.raw_positions:
            y_range, max_y_val, min_y_val, x_range, max_x_val, min_x_val = self.calculate_shake_metrics()
            max_shake = max(y_range, x_range)  # 取X和Y方向的最大抖动值
            print("\n=== 防抖振动测试最终结果 ===")
            print(f"总帧数: {self.frame_count}")
            print()
            print("按照防抖振动测试标准计算（同一像素点最高点与最低点的差量）:")
            
            # Y方向结果
            if max_y_val is not None:
                print(f"Y方向: 最低点={min_y_val:.0f}, 最高点={max_y_val:.0f}")
                print(f"Y方向像素点差量: {y_range:.1f} pixel ({max_y_val:.0f} - {min_y_val:.0f})")
            else:
                print(f"Y方向像素点差量: {y_range:.1f} pixel")
            
            # X方向结果
            if max_x_val is not None:
                print(f"X方向: 最左点={min_x_val:.0f}, 最右点={max_x_val:.0f}")
                print(f"X方向像素点差量: {x_range:.1f} pixel ({max_x_val:.0f} - {min_x_val:.0f})")
            else:
                print(f"X方向像素点差量: {x_range:.1f} pixel")
            
            print(f"最大振动值: {max_shake:.1f} pixel (取X、Y方向的最大值)")
            print(f"测试标准: ≤10 pixel")
            print(f"防抖振动测试结果: {'通过' if max_shake <= 10 else '失败'}")
            if max_shake > 10:
                print(f"超标: {max_shake - 10:.1f} pixel")
            
            # 保存测试结果到文件
            video_name = os.path.basename(self.video_path)
            self.save_test_result(video_name, y_range, max_y_val, min_y_val, x_range, max_x_val, min_x_val, max_shake, self.frame_count)
            
            # 等待用户按键后返回视频选择界面
            import msvcrt
            print("\n按任意键返回视频选择界面...")
            
            # 清空键盘缓冲区中的残留按键
            while msvcrt.kbhit():
                msvcrt.getch()
            
            # 等待用户实际按键
            while True:
                key = msvcrt.getch()
                if key:  # 任何有效按键都可以继续
                    break

def select_video_file():
    import os
    import msvcrt
    import sys
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for root, dirs, files in os.walk('.'):
        for f in files:
            if any(f.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, f))
    if not video_files:
        print("未找到任何视频文件！")
        input("按回车退出...")
        sys.exit(0)
    idx = 0
    
    # 只显示一次界面头部
    print("\n" + "="*60)
    print("请选择要分析的视频文件（↑↓选择，回车确定，q退出）：\n")
    
    def display_menu():
        for i, f in enumerate(video_files):
            prefix = '>>' if i == idx else '  '
            print(f"{prefix} {f}")
        print("\n按 'q' 退出程序")
    
    # 显示初始菜单
    display_menu()
    
    while True:
        key = msvcrt.getch()
        if key == b'\xe0':  # 方向键
            key2 = msvcrt.getch()
            if key2 == b'H':  # 上
                idx = (idx - 1) % len(video_files)
                # 向上移动光标到菜单开始位置并重新显示
                for _ in range(len(video_files) + 2):
                    print("\033[A", end="")
                display_menu()
            elif key2 == b'P':  # 下
                idx = (idx + 1) % len(video_files)
                # 向上移动光标到菜单开始位置并重新显示
                for _ in range(len(video_files) + 2):
                    print("\033[A", end="")
                display_menu()
        elif key == b'\r':  # 回车
            return video_files[idx]
        elif key == b'q':
            return None  # 返回 None 表示退出

def main():
    """主函数"""
    print("正在启动防抖振动测试程序...")
    print("初始化中，请稍候...")
    
    # 初始化全局测试结果管理器
    global_test_manager = AccurateShakeTest()
    global_test_manager.initialize_results_file()
    print("✓ 测试结果文件已初始化")
    
    while True:
        print("防抖振动测试程序（无滤波平滑）")
        print("=" * 60)
        print("正在扫描视频文件...")
        video_path = select_video_file()
        if video_path is None:  # 用户选择退出
            break
        print(f"已选择: {video_path}")
        print("正在初始化测试模块...")
        test = AccurateShakeTest(video_path)
        # 继承全局测试结果
        test.test_results = global_test_manager.test_results
        print("开始运行测试...")
        test.run_test()
        # 更新全局测试结果
        global_test_manager.test_results = test.test_results
        print("\n测试完成，返回文件选择界面...")
        print("=" * 60)

if __name__ == "__main__":
    try:
        print("\n启动主程序...")
        main()
    except Exception as e:
        print(f"\n程序出现错误: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
    finally:
        import msvcrt
        print("\n按任意键关闭窗口...")
        try:
            msvcrt.getch()
        except:
            input("按回车键关闭窗口...")