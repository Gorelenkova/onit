import os
import cv2
import numpy as np
import pytesseract
from pathlib import Path
import time
from dataclasses import dataclass


@dataclass
class Config:
    min_area: int = 1000
    max_area: int = 200000
    min_aspect: float = 0.2
    max_aspect: float = 10.0
    points_count: int = 128
    save_dir: Path = Path('./saved_rois')


cfg = Config()
cfg.save_dir.mkdir(exist_ok=True)


class ImageProcessor:
    @staticmethod
    def get_binary_mask(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    @staticmethod
    def filter_candidates(contours):
        valid = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if not (cfg.min_area <= area <= cfg.max_area):
                continue
            aspect = w / h if h != 0 else 0
            if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
                continue
            valid.append((cnt, x, y, w, h, area, aspect))
        valid.sort(key=lambda x: x[5], reverse=True)
        return valid

    @staticmethod
    def normalize_contour(contour, k=cfg.points_count):
        points = contour.reshape(-1, 2).astype(np.float32)
        distances = np.sqrt(((np.diff(points, axis=0, append=[points[0]])) ** 2).sum(axis=1))
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))
        total_len = cumulative[-1]

        if total_len == 0:
            return np.zeros(k, dtype=np.complex64)

        targets = np.linspace(0, total_len, num=k, endpoint=False)
        interpolated = []
        idx = 0

        for t in targets:
            while idx < len(cumulative) - 1 and cumulative[idx + 1] < t:
                idx += 1
            p1 = points[idx % len(points)]
            p2 = points[(idx + 1) % len(points)]
            seg_len = cumulative[idx + 1] - cumulative[idx] or 1e-6
            alpha = (t - cumulative[idx]) / seg_len
            interpolated.append((1 - alpha) * p1 + alpha * p2)

        interp_array = np.array(interpolated)
        diffs = np.diff(interp_array, axis=0, append=[interp_array[0]])
        return diffs[:, 0] + 1j * diffs[:, 1]

    @staticmethod
    def extract_text(roi):
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        h, w = roi.shape
        scaled = cv2.resize(roi, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(binary, lang='eng+rus')
        except:
            text = ''
        return text.strip(), binary


class ContourAnalyzer:
    @staticmethod
    def dot_product(a, b):
        return np.sum(a * np.conj(b))

    @staticmethod
    def vector_norm(v):
        return np.sqrt(np.sum(np.abs(v) ** 2))

    @staticmethod
    def autocorrelation(sequence):
        length = len(sequence)
        result = np.zeros(length, dtype=np.complex128)
        for shift in range(length):
            shifted = np.roll(sequence, -shift)
            result[shift] = ContourAnalyzer.dot_product(sequence, shifted)
        return result


class VideoProcessor:
    def __init__(self):
        self.camera = None
        self.frame_number = 0
        self.save_counter = 0
        self.text_history = []
        self.processor = ImageProcessor()
        self.analyzer = ContourAnalyzer()

    def start(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Ошибка: Камера не обнаружена")
            return False
        print("Система запущена. q - выход, s - сохранить")
        return True

    def draw_text_panel(self):
        panel = np.ones((600, 600, 3), dtype=np.uint8) * 255
        y_start = 20
        line_height = 22
        for i, line in enumerate(self.text_history[-25:]):
            cv2.putText(panel, line, (10, y_start + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return panel

    def process_frame(self, frame):
        display = frame.copy()
        mask = self.processor.get_binary_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = self.processor.filter_candidates(contours)

        for idx, (cnt, x, y, w, h, area, aspect) in enumerate(candidates[:6]):
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

            roi = frame[y1:y2, x1:x2]
            shifted_cnt = cnt - [x1, y1]
            normalized = self.processor.normalize_contour(shifted_cnt)
            acf = self.analyzer.autocorrelation(normalized)
            acf_max = np.max(np.abs(acf))
            norm_val = self.analyzer.vector_norm(normalized)

            detected_text, _ = self.processor.extract_text(roi)

            if detected_text:
                self.text_history.append(detected_text)
                if len(self.text_history) > 25:
                    self.text_history.pop(0)

            display_text = detected_text.replace('\n', ' | ')[:60]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, display_text, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            print(f"Кадр {self.frame_number} | Объект {idx} | Площадь:{area:.0f} "
                  f"Пропорции:{aspect:.2f} | Текст:'{detected_text[:30]}'")

        return display, candidates[:6]

    def save_regions(self, frame, regions):
        saved = 0
        for _, x, y, w, h, _, _ in regions:
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            roi = frame[y1:y2, x1:x2]
            filename = cfg.save_dir / f'region_{int(time.time())}_{self.save_counter}.png'
            cv2.imwrite(str(filename), roi)
            self.save_counter += 1
            saved += 1
        print(f"Сохранено областей: {saved}")

    def run(self):
        if not self.start():
            return

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Ошибка захвата кадра")
                break

            processed_frame, regions = self.process_frame(frame)
            text_panel = self.draw_text_panel()

            cv2.imshow("Распознавание объектов", processed_frame)
            cv2.imshow("Распознанный текст", text_panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_regions(frame, regions)

            self.frame_number += 1

        self.camera.release()
        cv2.destroyAllWindows()


def main():
    app = VideoProcessor()
    app.run()


if __name__ == '__main__':
    main()