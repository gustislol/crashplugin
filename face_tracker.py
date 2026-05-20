import cv2
import requests
import numpy as np
import time
import threading
import os
import json
import base64
import zipfile
from urllib.request import urlopen
from pathlib import Path
from typing import List

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

CAMERA_IP = "192.168.0.31"
SERVO_IP = "192.168.0.30"
CAMERA_PORT = 8000
SERVO_PORT = 8001
CAMERA_URL = f"http://{CAMERA_IP}:{CAMERA_PORT}/frame"
SERVO_URL = f"http://{SERVO_IP}:{SERVO_PORT}/servo"
DATASET_DIR = Path("training_photos")
ROBOFLOW_ZIP_DIR = Path("training_photos/roboflow_zip")
ROBOFLOW_EXTRACT_DIR = Path("training_photos/roboflow_extracted")
EXTRA_ZIP_DIRS_ENV = "ROBOFLOW_ZIP_EXTRA_DIRS"
DEFAULT_WINDOWS_ZIP_DIR = Path(r"C:/Users/marie/Downloads/yolov8")
SERVO_X_MIN, SERVO_X_MAX = 0, 180
SERVO_Y_MIN, SERVO_Y_MAX = 0, 180


class GeminiTracker:
    def __init__(self, api_key: str, width: int = 320, height: int = 240):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai niet geïnstalleerd")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.lock_hint = ""
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._result = None
        self._busy = False
        self._last_called = 0.0
        self._call_interval = 0.12

    def _frame_to_b64(self, frame: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode()

    def _call_gemini(self, frame: np.ndarray):
        try:
            b64 = self._frame_to_b64(frame)
            prompt = (
                f"Frame is {self.width}x{self.height}. You are a face tracking model. "
                f"Track the SAME person across frames. Lock hint: {self.lock_hint}. "
                "Prefer temporal consistency over switching person. Return ONLY JSON: "
                '{"cx":int|null,"cy":int|null,"w":int|null,"h":int|null,'
                '"confidence":float,"label":string,"direction":"LEFT|CENTER|RIGHT"}'
            )
            resp = self.model.generate_content([
                {"mime_type": "image/jpeg", "data": b64},
                prompt,
            ])
            text = resp.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            cx, cy = data.get("cx"), data.get("cy")
            if cx is None or cy is None:
                result = None
            else:
                w = int(data.get("w") or 64)
                h = int(data.get("h") or 64)
                conf = float(data.get("confidence", 0.0))
                direction = str(data.get("direction", "CENTER")).upper()
                result = (
                    max(0, min(self.width - 1, int(cx))),
                    max(0, min(self.height - 1, int(cy))),
                    max(24, min(self.width, w)),
                    max(24, min(self.height, h)),
                    max(0.0, min(1.0, conf)),
                    str(data.get("label", "")),
                    direction,
                )
            with self._lock:
                self._result = result
        except Exception:
            pass
        finally:
            with self._lock:
                self._busy = False

    def submit_frame(self, frame: np.ndarray, lock_hint: str = ""):
        self.lock_hint = lock_hint
        now = time.time()
        with self._lock:
            if self._busy:
                return
        if now - self._last_called < self._call_interval:
            return
        self._last_called = now
        with self._lock:
            self._busy = True
        threading.Thread(target=self._call_gemini, args=(frame.copy(),), daemon=True).start()

    def get_result(self):
        with self._lock:
            return self._result


class FaceTracker:
    def __init__(self, gemini_key: str = ""):
        self.width, self.height = 320, 240
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.servo_x = 90.0
        self.servo_y = 90.0
        self.servo_x_min = int(os.environ.get("SERVO_X_MIN", SERVO_X_MIN))
        self.servo_x_max = int(os.environ.get("SERVO_X_MAX", SERVO_X_MAX))
        self.servo_y_min = int(os.environ.get("SERVO_Y_MIN", SERVO_Y_MIN))
        self.servo_y_max = int(os.environ.get("SERVO_Y_MAX", SERVO_Y_MAX))
        self.servo_x_vel = 0.0
        self.servo_y_vel = 0.0

        # Strong, stable controller
        self.kp = 0.018
        self.kd = 0.007
        self.max_step = 4.0
        self.dead_zone = 6
        self.last_error_x = 0.0
        self.last_error_y = 0.0

        # Target lock state
        self.locked_face = None
        self.lock_lost_frames = 0
        self.max_lock_lost = 28
        self.reacquire_hold = 0
        self.reacquire_hold_max = 24
        self.ai_direction = "CENTER"
        self.gemini_conf = 0.0

        # Smoothed target point
        self.target_x = float(self.center_x)
        self.target_y = float(self.center_y)
        self.target_alpha = 0.52
        self.max_predict_frames = 12
        self.predict_frames = 0
        self.target_vx = 0.0
        self.target_vy = 0.0

        # Between-detection visual tracker for much more stable lock
        self.cv_tracker = None
        self.cv_tracker_ok = False
        self.cv_tracker_age = 0
        self.cv_tracker_max_age = 18

        self.latest_frame = None
        self.prev_frame = None
        self.frame_lock = threading.Lock()
        self.fps = 0.0
        self.fps_t = time.time()
        self.fps_counter = 0

        self.last_sent_x = None
        self.last_sent_y = None
        self.last_send_t = 0.0
        self.min_send_interval = 0.015
        self.min_delta_send = 0

        self.gemini = None
        if gemini_key and GEMINI_AVAILABLE:
            try:
                self.gemini = GeminiTracker(gemini_key, self.width, self.height)
            except Exception:
                self.gemini = None

        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

                # Generic face profile learning (drop photos in training_photos/)
        self.profile_hist = None
        self.profile_ready = False
        self.profile_score = 0.0
        self.dataset_count = 0
        self.profile_loading = True
        self._prepare_roboflow_dataset()
        self.profile_thread = threading.Thread(target=self._load_person_profile, daemon=True)
        self.profile_thread.start()


    def _collect_zip_dirs(self) -> List[Path]:
        dirs = [ROBOFLOW_ZIP_DIR]
        extra = os.environ.get(EXTRA_ZIP_DIRS_ENV, "").strip()
        if extra:
            for part in extra.split(";"):
                part = part.strip().strip('"')
                if part:
                    dirs.append(Path(part))
        dirs.append(DEFAULT_WINDOWS_ZIP_DIR)

        unique = []
        seen = set()
        for d in dirs:
            key = str(d).lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)
        return unique

    def _prepare_roboflow_dataset(self):
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        ROBOFLOW_ZIP_DIR.mkdir(parents=True, exist_ok=True)
        ROBOFLOW_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

        zip_dirs = self._collect_zip_dirs()
        zips = []
        for zd in zip_dirs:
            try:
                if zd.exists() and zd.is_dir():
                    zips.extend(sorted(zd.rglob("*.zip")))
            except Exception:
                pass
        # dedupe zip paths
        uniq = []
        seen = set()
        for z in zips:
            k = str(z).lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(z)

        for z in uniq:
            target = ROBOFLOW_EXTRACT_DIR / z.stem
            stamp = target / ".unzipped.ok"
            try:
                if stamp.exists() and stamp.stat().st_mtime >= z.stat().st_mtime:
                    continue
                if target.exists():
                    # refresh extracted files if zip changed
                    for f in target.rglob("*"):
                        if f.is_file():
                            f.unlink()
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(target)
                stamp.write_text("ok", encoding="utf-8")
            except Exception:
                pass

    def _load_person_profile(self):
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [f for f in DATASET_DIR.iterdir() if f.is_file() and f.suffix.lower() in exts]
        for root in ROBOFLOW_EXTRACT_DIR.glob("*"):
            if root.is_dir():
                files.extend([f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in exts])
        files = list(dict.fromkeys(files))
        if not files:
            return

        hists = []
        for img_path in files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Try face detect on training photo; fallback to center crop
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                roi = img[y:y+h, x:x+w]
            else:
                h0, w0 = img.shape[:2]
                cw, ch = int(w0 * 0.5), int(h0 * 0.5)
                cx, cy = w0 // 2, h0 // 2
                x1 = max(0, cx - cw // 2)
                y1 = max(0, cy - ch // 2)
                roi = img[y1:y1+ch, x1:x1+cw]

            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (96, 96))
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hists.append(hist)

        if hists:
            self.profile_hist = np.mean(np.array(hists), axis=0)
            self.profile_ready = True
            self.dataset_count = len(hists)
        self.profile_loading = False

    def _profile_match_score(self, frame, rect):
        if not self.profile_ready:
            return 0.0
        x, y, w, h = rect
        x = max(0, min(frame.shape[1]-1, x))
        y = max(0, min(frame.shape[0]-1, y))
        w = max(10, min(frame.shape[1]-x, w))
        h = max(10, min(frame.shape[0]-y, h))
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0
        roi = cv2.resize(roi, (96, 96))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        d = cv2.compareHist(self.profile_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
        return float(max(0.0, 1.0 - d))

    def get_frame(self):
        try:
            resp = urlopen(CAMERA_URL, timeout=0.35)
            arr = np.frombuffer(resp.read(), np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _camera_loop(self):
        while True:
            frame = self.get_frame()
            if frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def get_latest_frame(self):
        with self.frame_lock:
            frame = self.latest_frame
        if frame is None:
            return None
        den = cv2.fastNlMeansDenoisingColored(frame, None, 4, 4, 5, 15)
        blur = cv2.GaussianBlur(den, (0, 0), 1.0)
        enhanced = cv2.addWeighted(den, 1.28, blur, -0.28, 0)
        if self.prev_frame is None or self.prev_frame.shape != enhanced.shape:
            self.prev_frame = enhanced
            return enhanced
        stable = cv2.addWeighted(enhanced, 0.82, self.prev_frame, 0.18, 0)
        self.prev_frame = enhanced
        return stable

    def _iou(self, a, b):
        if a is None or b is None:
            return 0.0
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / max(1, union)

    def _center_distance(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        acx, acy = ax + aw / 2, ay + ah / 2
        bcx, bcy = bx + bw / 2, by + bh / 2
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    def _select_locked_face(self, faces, frame):
        if not faces:
            return None

        if self.locked_face is None:
            if self.profile_ready:
                return max(faces, key=lambda f: self._profile_match_score(frame, f))
            return max(faces, key=lambda f: f[2] * f[3])

        # Score = overlap priority + distance continuity + size continuity
        lx, ly, lw, lh = self.locked_face
        best = None
        best_score = -10**9
        for f in faces:
            iou = self._iou(f, self.locked_face)
            dist = self._center_distance(f, self.locked_face)
            size_ratio = min(f[2] * f[3], lw * lh) / max(1, max(f[2] * f[3], lw * lh))
            profile = self._profile_match_score(frame, f)
            score = (iou * 2.9) + (size_ratio * 1.1) - (dist / 220.0) + (profile * 1.2)
            if score > best_score:
                best_score = score
                best = f

        if best_score < -0.25:
            return None
        return best

    def _reset_cv_tracker(self):
        self.cv_tracker = None
        self.cv_tracker_ok = False
        self.cv_tracker_age = 0

    def _start_cv_tracker(self, frame, rect):
        x, y, w, h = [int(v) for v in rect]
        if w <= 0 or h <= 0:
            return
        try:
            self.cv_tracker = cv2.TrackerCSRT_create()
            self.cv_tracker_ok = self.cv_tracker.init(frame, (x, y, w, h))
            self.cv_tracker_age = 0
        except Exception:
            self._reset_cv_tracker()

    def _update_cv_tracker(self, frame):
        if self.cv_tracker is None:
            return None
        try:
            ok, box = self.cv_tracker.update(frame)
            if not ok:
                return None
            x, y, w, h = [int(v) for v in box]
            if w < 16 or h < 16:
                return None
            self.cv_tracker_age += 1
            return (x, y, w, h)
        except Exception:
            return None

    def find_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        faces = []
        for sf, neigh, msz in [(1.03, 3, (18, 18)), (1.06, 4, (22, 22)), (1.10, 5, (28, 28))]:
            d = self.face_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=neigh, minSize=msz)
            if len(d) > 0:
                faces = [tuple(map(int, f)) for f in d]
                break

        if self.gemini:
            lock_hint = "none"
            if self.locked_face is not None:
                lx, ly, lw, lh = self.locked_face
                lock_hint = f"x={lx},y={ly},w={lw},h={lh}"
            self.gemini.submit_frame(frame, lock_hint=lock_hint)
        g = self.gemini.get_result() if self.gemini else None

        selected = self._select_locked_face(faces, frame)

        # if detector misses, try visual tracker continuity
        if selected is None:
            tbox = self._update_cv_tracker(frame)
            if tbox is not None and self.cv_tracker_age <= self.cv_tracker_max_age:
                selected = tbox

        if selected is None:
            # bootstrap lock from Gemini when detector is weak in low light
            if g and g[4] >= 0.25 and self.locked_face is None:
                gx, gy, gw, gh, conf, _lbl, direction = g
                self.gemini_conf = conf
                self.ai_direction = direction
                self.locked_face = (max(0, gx - gw // 2), max(0, gy - gh // 2), gw, gh)
                self._start_cv_tracker(frame, self.locked_face)
                self.target_x, self.target_y = float(gx), float(gy)
                return int(self.target_x), int(self.target_y), self.locked_face

            self.lock_lost_frames += 1
            if g and g[4] >= 0.32:
                gx, gy, gw, gh, conf, _lbl, direction = g
                self.gemini_conf = conf
                self.ai_direction = direction
                self.locked_face = (max(0, gx - gw // 2), max(0, gy - gh // 2), gw, gh)
                self.target_x = 0.65 * self.target_x + 0.35 * gx
                self.target_y = 0.65 * self.target_y + 0.35 * gy
                self.predict_frames = 0
                return int(self.target_x), int(self.target_y), self.locked_face

            if self.lock_lost_frames <= self.max_predict_frames:
                self.predict_frames += 1
                self.target_x += self.target_vx
                self.target_y += self.target_vy
                self.target_x = float(np.clip(self.target_x, 0, self.width - 1))
                self.target_y = float(np.clip(self.target_y, 0, self.height - 1))
                return int(self.target_x), int(self.target_y), self.locked_face

            if self.lock_lost_frames > self.max_lock_lost:
                self.locked_face = None
                self.reacquire_hold = self.reacquire_hold_max
                self._reset_cv_tracker()
            return None, None, None

        self.lock_lost_frames = 0
        self.locked_face = selected
        self.cv_tracker_age = 0
        if self.cv_tracker is None or not self.cv_tracker_ok:
            self._start_cv_tracker(frame, selected)
        self.profile_score = self._profile_match_score(frame, selected) if self.profile_ready else 0.0

        x, y, w, h = selected
        hx, hy = x + w // 2, y + h // 2
        tx, ty = float(hx), float(hy)

        if g:
            gx, gy, gw, gh, conf, _lbl, direction = g
            self.gemini_conf = conf
            self.ai_direction = direction

            # AI heavily weighted but bounded by lock continuity
            if conf >= 0.35:
                ai_w = 0.55 + 0.40 * conf
                tx = ai_w * gx + (1.0 - ai_w) * hx
                ty = ai_w * gy + (1.0 - ai_w) * hy

                ax = max(0, min(self.width - gw, gx - gw // 2))
                ay = max(0, min(self.height - gh, gy - gh // 2))
                ai_rect = (ax, ay, gw, gh)
                if self._iou(ai_rect, selected) > 0.03 or conf > 0.75:
                    x = int(0.28 * x + 0.72 * ai_rect[0])
                    y = int(0.28 * y + 0.72 * ai_rect[1])
                    w = int(0.32 * w + 0.68 * ai_rect[2])
                    h = int(0.32 * h + 0.68 * ai_rect[3])
                    selected = (x, y, w, h)
                    self.locked_face = selected
                    self._start_cv_tracker(frame, selected)
        else:
            self.ai_direction = "CENTER"
            self.gemini_conf = 0.0

        # Exponential smoothing on target point
        old_x, old_y = self.target_x, self.target_y
        self.target_x = self.target_alpha * self.target_x + (1.0 - self.target_alpha) * tx
        self.target_y = self.target_alpha * self.target_y + (1.0 - self.target_alpha) * ty
        self.target_vx = 0.70 * self.target_vx + 0.30 * (self.target_x - old_x)
        self.target_vy = 0.70 * self.target_vy + 0.30 * (self.target_y - old_y)
        self.predict_frames = 0

        return int(self.target_x), int(self.target_y), selected

    def update_servo(self, face_x, face_y):
        if face_x is not None and face_y is not None:
            error_x = face_x - self.center_x
            error_y = face_y - self.center_y
            if abs(error_x) < self.dead_zone:
                error_x = 0
            if abs(error_y) < self.dead_zone:
                error_y = 0

            dx = error_x - self.last_error_x
            dy = error_y - self.last_error_y
            self.last_error_x = error_x
            self.last_error_y = error_y

            # adaptive step limit: faster on big error, gentle near center
            dyn_max = float(np.clip(2.2 + 0.03 * max(abs(error_x), abs(error_y)), 2.2, self.max_step))
            step_x = np.clip(self.kp * error_x + self.kd * dx + 0.0018 * error_x, -dyn_max, dyn_max)
            step_y = np.clip(self.kp * error_y + self.kd * dy + 0.0018 * error_y, -dyn_max, dyn_max)

            self.servo_x_vel = 0.64 * self.servo_x_vel + 0.36 * step_x
            self.servo_y_vel = 0.64 * self.servo_y_vel + 0.36 * step_y
            self.servo_x += self.servo_x_vel
            self.servo_y += self.servo_y_vel

            # If hitting boundary, avoid getting stuck with wrong velocity sign
            if self.servo_x <= self.servo_x_min + 0.5 and self.servo_x_vel < 0:
                self.servo_x_vel *= 0.2
            if self.servo_x >= self.servo_x_max - 0.5 and self.servo_x_vel > 0:
                self.servo_x_vel *= 0.2
            if self.servo_y <= self.servo_y_min + 0.5 and self.servo_y_vel < 0:
                self.servo_y_vel *= 0.2
            if self.servo_y >= self.servo_y_max - 0.5 and self.servo_y_vel > 0:
                self.servo_y_vel *= 0.2
        else:
            # no face: damp movement quickly so servo does not drift
            self.servo_x_vel *= 0.78
            self.servo_y_vel *= 0.78

        self.servo_x = float(np.clip(self.servo_x, self.servo_x_min, self.servo_x_max))
        self.servo_y = float(np.clip(self.servo_y, self.servo_y_min, self.servo_y_max))
        return int(round(self.servo_x)), int(round(self.servo_y))

    def send_servo(self, x, y):
        now = time.time()
        if self.last_sent_x is not None and self.last_sent_y is not None:
            # skip only exact duplicates (or configured threshold)
            if abs(x - self.last_sent_x) <= self.min_delta_send and abs(y - self.last_sent_y) <= self.min_delta_send:
                return True
            # allow urgent sends when movement jump is bigger
            jump = max(abs(x - self.last_sent_x), abs(y - self.last_sent_y))
            if now - self.last_send_t < self.min_send_interval and jump < 3:
                return True

        try:
            r = requests.get(f"{SERVO_URL}?x={x}&y={y}", timeout=2)
            ok = r.status_code == 200
            if not ok:
                # fallback for firmware that only supports x
                r2 = requests.get(f"{SERVO_URL}?x={x}", timeout=2)
                ok = r2.status_code == 200
            if ok:
                self.last_sent_x = x
                self.last_sent_y = y
                self.last_send_t = now
            return ok
        except Exception:
            return False

    def draw(self, frame, rect, sx, sy):
        img = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_CUBIC)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (960, 58), (18, 20, 30), -1)
        cv2.rectangle(overlay, (0, 662), (960, 720), (18, 20, 30), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

        if rect is not None:
            x, y, w, h = rect
            x *= 3
            y *= 3
            w *= 3
            h *= 3
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 140), 2)
            cv2.circle(img, (x + w // 2, y + h // 2), 4, (0, 255, 140), -1)

        cv2.putText(img, "FACE TRACKER PRO MAX", (16, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        cv2.putText(img, f"FPS {self.fps:.1f}", (818, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 200), 2)

        lock_txt = "LOCKED" if rect is not None else "SEARCHING"
        lock_col = (0, 255, 120) if rect is not None else (0, 165, 255)
        cv2.putText(img, f"TARGET: {lock_txt}", (16, 696), cv2.FONT_HERSHEY_SIMPLEX, 0.78, lock_col, 2)
        cv2.putText(img, f"AI POS: {self.ai_direction}", (320, 696), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (80, 220, 255), 2)
        cv2.putText(img, f"AI CONF: {self.gemini_conf:.2f}", (560, 696), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (80, 220, 255), 2)
        if self.profile_loading:
            prof_txt = "DATASET: loading... (tracking works already)"
        elif self.profile_ready:
            prof_txt = f"DATASET-MATCH: {self.profile_score:.2f}  n={self.dataset_count}"
        else:
            prof_txt = "DATASET-MATCH: add photos to training_photos/"
        cv2.putText(img, prof_txt, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        cv2.putText(img, f"SERVO X:{sx} Y:{sy}", (700, 696), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 2)
        age_ms = int((time.time() - self.last_send_t) * 1000) if self.last_send_t else -1
        cv2.putText(img, f"SEND_AGE:{age_ms}ms", (700, 668), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
        cv2.putText(img, f"LIM X[{self.servo_x_min},{self.servo_x_max}] Y[{self.servo_y_min},{self.servo_y_max}]", (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 255), 1)
        return img

    def run(self):
        cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Tracker", 960, 720)

        while True:
            frame = self.get_latest_frame()
            if frame is None:
                waiting = np.zeros((720, 960, 3), dtype=np.uint8)
                cv2.putText(waiting, "Connecting camera...", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
                cv2.imshow("Face Tracker", waiting)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
                time.sleep(0.01)
                continue

            self.fps_counter += 1
            now = time.time()
            if now - self.fps_t >= 0.5:
                self.fps = self.fps_counter / (now - self.fps_t)
                self.fps_counter = 0

                self.fps_t = now

            fx, fy, rect = self.find_face(frame)
            sx, sy = self.update_servo(fx, fy)
            self.send_servo(sx, sy)

            view = self.draw(frame, rect, sx, sy)
            cv2.imshow("Face Tracker", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    FaceTracker(key).run()
