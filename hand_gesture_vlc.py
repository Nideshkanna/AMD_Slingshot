import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import os
import math
import time
from collections import deque
import threading
try:
    import pygetwindow as gw
except Exception:
    gw = None


class HandGestureVLC:
    def __init__(self, model_path="models/hand_landmarker.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        # Configure hand detector with detection thresholds
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
            min_tracking_confidence=0.25
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # State
        self.mode = "MEDIA"
        self.media_state = "PLAYING"
        self.current_gesture = "NONE"
        
        # Gesture detection histories for smoothing
        self.gesture_history = deque(maxlen=5)
        self.pinch_history = deque(maxlen=3)
        self.palm_history = deque(maxlen=8)
        self.thumb_right_count = 0
        self.thumb_left_count = 0
        self.last_action_time = self.last_volume_time = self.last_swipe_time = self.mute_cooldown = 0
        self.last_track_action_time = 0 
        self.last_volume_y = None
        self.mute_lock = self.last_swipe_direction = self.swipe_display_time = None
        self.mute_action_time = 0 
        
        # Performance tracking
        self.prev_frame_time = 0
        self.fps = 0
        self.latency = 0
        self.target_fps = 20
        self.frame_skip = 0
        
        # Hand size caching to avoid recalculation
        self.hand_size = 0
        self.hand_size_timestamp = 0
        self.hand_size_cache_duration = 0.5

        # Volume control state
        self.volume_direction = None
        self.volume_direction_start = None
        self.volume_base_interval = 0.15
        
        # Asynchronous frame capture
        self.frame_queue = deque(maxlen=1)
        self.running = True
        self.camera_thread = None

    def dist(self, a, b):
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt(dx * dx + dy * dy)
    
    def get_hand_size(self, lm, now):
        """Get cached hand size with periodic refresh"""
        if now - self.hand_size_timestamp > self.hand_size_cache_duration or self.hand_size == 0:
            self.hand_size = self.dist(lm[0], lm[12])
            self.hand_size_timestamp = now
        return self.hand_size

    def is_extended(self, tip, pip, lm):
        return lm[tip].y < lm[pip].y

    def is_closed(self, tip, pip, lm):
        return lm[tip].y > lm[pip].y

    def detect_open_palm(self, lm):
        return sum(self.is_extended(t, p, lm) for t, p in zip([4, 8, 12, 16, 20], [2, 6, 10, 14, 18])) >= 4

    def detect_closed_fist(self, lm):
        all_fingers = [4, 8, 12, 16, 20]
        pip_fingers = [2, 6, 10, 14, 18]
        closed_count = sum(self.is_closed(t, p, lm) for t, p in zip(all_fingers, pip_fingers))
        return closed_count >= 4

    def detect_thumb_right(self, lm):
        thumb_extended = self.is_extended(4, 2, lm)
        thumb_right = lm[4].x > lm[2].x + 0.06
        other_fingers_closed = sum(self.is_closed(t, p, lm) for t, p in zip([8, 12, 16, 20], [6, 10, 14, 18])) >= 4
        return thumb_extended and thumb_right and other_fingers_closed

    def detect_thumb_left(self, lm):
        thumb_extended = self.is_extended(4, 2, lm)
        thumb_left = lm[4].x < lm[2].x - 0.06
        other_fingers_closed = sum(self.is_closed(t, p, lm) for t, p in zip([8, 12, 16, 20], [6, 10, 14, 18])) >= 4
        return thumb_extended and thumb_left and other_fingers_closed

    def detect_pinch(self, lm):
        return self.dist(lm[4], lm[8]) < 0.080 and self.is_extended(12, 10, lm)

    def detect_mute(self, lm, now):
        """Detect mute gesture (three fingers extended)"""
        hand_size = self.get_hand_size(lm, now)
        ext_th = max(0.08, hand_size * 0.5)
        cls_th = max(0.06, hand_size * 0.35)
        
        # Check if key fingers are extended
        if not (self.is_extended(4, 2, lm) and self.is_extended(8, 6, lm)):
            return False
        
        # Calculate finger distances
        thumb_dist = self.dist(lm[4], lm[2])
        index_dist = self.dist(lm[8], lm[5])
        pinky_dist = self.dist(lm[20], lm[17])
        
        return (thumb_dist > ext_th and index_dist > ext_th and pinky_dist > ext_th and
                lm[12].y - lm[10].y > cls_th and lm[12].y > lm[9].y and
                lm[16].y - lm[14].y > cls_th and lm[16].y > lm[13].y)

    def is_vlc_focused(self):
        if gw is None: return True
        try:
            active_window = gw.getActiveWindow()
            if active_window is None: return False
            window_title = active_window.title.lower()
            return "vlc" in window_title
        except: return False

    def is_stable(self, gesture, threshold):
        if not self.gesture_history: return False
        return sum(1 for g in self.gesture_history if g == gesture) >= threshold

    def action(self, act, now):
        if not self.is_vlc_focused(): return
        if act == "PLAY" and now - self.last_action_time > 0.3:
            if self.media_state != "PLAYING":
                pyautogui.press("space")
                self.media_state = "PLAYING"
                self.last_action_time = now
        elif act == "PAUSE" and now - self.last_action_time > 0.3:
            if self.media_state != "PAUSED":
                pyautogui.press("space")
                self.media_state = "PAUSED"
                self.last_action_time = now
        elif act == "MUTE" and now >= self.mute_cooldown and not self.mute_lock:
            pyautogui.press("m")
            self.mute_lock = True
            self.mute_cooldown = now + 0.5
            self.mute_action_time = now 
        elif act == "NEXT" and now - self.last_track_action_time > 2.0:
            pyautogui.press("n")
            self.media_state = "PLAYING"
            self.last_swipe_direction = "NEXT ➡️"
            self.swipe_display_time = now + 1.0
            self.last_track_action_time = now
        elif act == "PREVIOUS" and now - self.last_track_action_time > 2.0:
            pyautogui.press("p")
            self.media_state = "PLAYING"
            self.last_swipe_direction = "PREVIOUS ⬅️"
            self.swipe_display_time = now + 1.0
            self.last_track_action_time = now

    def volume(self, lm, h, now):
        thumb_y = lm[4].y * h
        if self.last_volume_y is None:
            self.last_volume_y = thumb_y
            return
        
        if self.volume_direction is not None:
            hold_duration = now - self.volume_direction_start
            interval = self.volume_base_interval if hold_duration < 2.0 else self.volume_base_interval / 2
            if now - self.last_volume_time >= interval:
                pyautogui.hotkey("ctrl", "up") if self.volume_direction == 'UP' else pyautogui.hotkey("ctrl", "down")
                self.last_volume_time = now
        
        delta = self.last_volume_y - thumb_y
        current_direction = 'UP' if delta > 15 else 'DOWN' if delta < -15 else None
        if current_direction and current_direction != self.volume_direction:
            self.volume_direction, self.volume_direction_start, self.last_volume_time = current_direction, now, now - self.volume_base_interval
        self.last_volume_y = thumb_y

    def _capture_frames(self, cap):
        """Capture frames asynchronously in background thread"""
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_queue.append((ret, frame))

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start background frame capture thread
        self.camera_thread = threading.Thread(target=self._capture_frames, args=(cap,), daemon=True)
        self.camera_thread.start()

        frame_count = 0
        target_frame_skip = 1  # Process every frame initially, tune if needed

        while True:
            # Retrieve latest frame from queue
            if not self.frame_queue:
                time.sleep(0.001)
                continue
            
            ret, frame = self.frame_queue[0]
            if not ret:
                break

            frame_count += 1
            
            # Skip frames if needed
            if frame_count % target_frame_skip != 0:
                continue

            start_time = time.time()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Resize frame for inference
            small_frame = cv2.resize(frame, (224, 168), interpolation=cv2.INTER_LINEAR)
            
            # Convert to RGB for model
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Run hand detection
            timestamp_ms = int(time.time() * 1000)
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            
            self.latency = (time.time() - start_time) * 1000
            now = time.time()

            if not result.hand_landmarks:
                self.current_gesture = "NONE"
                self.gesture_history.clear()
                self.volume_direction = None
                if self.mode == "VOLUME" and now - getattr(self, '_pinch_exit', 0) > 0.3:
                    self.mode, self.last_volume_y = "MEDIA", None
            else:
                lm = result.hand_landmarks[0]
                is_pinch = self.detect_pinch(lm)
                
                if is_pinch:
                    self.pinch_history.append(True)
                    self._pinch_exit = now
                    if sum(self.pinch_history) >= 2 and self.mode != "VOLUME":
                        self.mode, self.last_volume_y = "VOLUME", None
                else:
                    self.pinch_history.append(False)
                    if sum(1 for p in self.pinch_history if not p) >= 1 and self.mode == "VOLUME":
                        self.mode, self.last_volume_y, self.volume_direction = "MEDIA", None, None
                        self.gesture_history.clear()

                if self.mode == "VOLUME" and is_pinch:
                    self.current_gesture = "PINCH"
                    self.volume(lm, h, now)
                else:
                    if self.detect_mute(lm, now):
                        self.current_gesture = "MUTE"
                        self.action("MUTE", now)
                    else:
                        self.mute_lock = False
                        if now - self.mute_action_time < 0.6:
                            self.gesture_history.clear()
                            self.current_gesture = "NEUTRAL"
                        elif self.detect_thumb_right(lm):
                            self.thumb_right_count += 1
                            self.current_gesture = "THUMB_RIGHT"
                            if self.thumb_right_count == 2:
                                self.action("NEXT", now)
                                self.swipe_display_time, self.thumb_right_count = now + 1.0, 0
                        elif self.detect_thumb_left(lm):
                            self.thumb_left_count += 1
                            self.current_gesture = "THUMB_LEFT"
                            if self.thumb_left_count == 2:
                                self.action("PREVIOUS", now)
                                self.swipe_display_time, self.thumb_left_count = now + 1.0, 0
                        elif self.detect_closed_fist(lm):
                            self.gesture_history.append("CLOSED_FIST")
                            self.current_gesture = "CLOSED_FIST"
                        elif self.detect_open_palm(lm):
                            self.gesture_history.append("OPEN_PALM")
                            self.current_gesture = "OPEN_PALM"
                        else:
                            self.gesture_history.append("NEUTRAL")
                            self.current_gesture = "NEUTRAL"
                        
                        # Check gesture stability for confirmation
                        if len(self.gesture_history) >= 4:
                            if self.is_stable("OPEN_PALM", 3):
                                self.action("PLAY", now)
                                self.gesture_history.clear()
                            elif self.is_stable("CLOSED_FIST", 3):
                                self.action("PAUSE", now)
                                self.gesture_history.clear()

                # Draw hand landmarks on frame
                key_points = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
                for idx in key_points:
                    p = lm[idx]
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)

            # Calculate and display performance metrics
            end_time = time.time()
            frame_duration = end_time - start_time
            
            self.fps = 1 / frame_duration if frame_duration > 0 else 0
            
            # Regulate frame rate
            target_duration = 1.0 / self.target_fps
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

            # Display HUD overlay
            cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"LAT: {int(self.latency)}ms", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"MODE: {self.mode}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if self.last_swipe_direction and now < self.swipe_display_time:
                cv2.putText(frame, self.last_swipe_direction, (w//2-50, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Gesture VLC Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.running = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    HandGestureVLC("models/hand_landmarker.task").run() 
