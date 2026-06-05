"""
app.py — Ứng dụng phân loại cảm xúc văn bản bằng mô hình Bidirectional LSTM
Stack : Python 3.x | Tkinter | TensorFlow/Keras | NumPy | Pickle
Author: Nhóm 2
"""

import os
import pickle
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "best_model_balanced.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
ENCODER_PATH   = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
MAX_LEN        = 100   # độ dài sequence đã dùng khi train

# Màu sắc giao diện
COLORS = {
    "bg_dark"    : "#0f1117",
    "bg_card"    : "#1a1d27",
    "bg_input"   : "#252836",
    "accent"     : "#6c63ff",
    "accent2"    : "#ff6584",
    "text_main"  : "#e2e8f0",
    "text_muted" : "#8892a4",
    "border"     : "#2d3148",
    "positive"   : "#22c55e",
    "negative"   : "#ef4444",
    "neutral"    : "#f59e0b",
    "btn_hover"  : "#7c73ff",
}

# Mapping nhãn: index → tên hiển thị
LABEL_MAP = {
    0: "Tích cực 😊",
    1: "Tiêu cực 😞",
    2: "Trung lập 😐",
}

LABEL_COLOR = {
    0: COLORS["positive"],
    1: COLORS["negative"],
    2: COLORS["neutral"],
}


# ─────────────────────────────────────────────────────────────────────────────
# HÀM TẢI TÀI NGUYÊN
# ─────────────────────────────────────────────────────────────────────────────
def load_resources():
    """
    Tải model LSTM, tokenizer và label_encoder từ đĩa.
    Trả về tuple (model, tokenizer, label_encoder_dict).
    Ném FileNotFoundError nếu thiếu file.
    Sử dụng custom patched layers để tương thích với Keras 3.x.
    """
    # Kiểm tra sự tồn tại của file
    for path, name in [(MODEL_PATH, "Model"), (TOKENIZER_PATH, "Tokenizer")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} không tìm thấy tại:\n{path}\n\n"
                "Hãy đảm bảo file nằm cùng thư mục với app.py."
            )

    # ── 1. Load tokenizer ──────────────────────────────────────────────────
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # ── 2. Load label encoder ──────────────────────────────────────────────
    label_encoder = None
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

    # ── 3. Load model với patched layers (workaround Keras 3 compatibility) ─
    import keras

    class _PatchedEmbedding(keras.layers.Embedding):
        """Bỏ qua tham số quantization_config không được nhận diện trong Keras 3."""
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    class _PatchedDense(keras.layers.Dense):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    class _PatchedLSTM(keras.layers.LSTM):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    class _PatchedBidirectional(keras.layers.Bidirectional):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    custom_objs = {
        "Embedding"     : _PatchedEmbedding,
        "Dense"         : _PatchedDense,
        "LSTM"          : _PatchedLSTM,
        "Bidirectional" : _PatchedBidirectional,
    }

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects=custom_objs,
    )

    return model, tokenizer, label_encoder


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str, tokenizer):
    """
    Chuyển chuỗi văn bản thô thành mảng numpy đã padding sẵn sàng cho model.
    Bước 1: texts_to_sequences  → list of int tokens
    Bước 2: pad_sequences       → shape (1, MAX_LEN)
    Trả về numpy array shape (1, MAX_LEN).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences([text])
    padded    = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, model, tokenizer, label_encoder):
    """
    Nhận chuỗi văn bản, chạy qua pipeline preprocess → model.predict → decode.
    Trả về tuple (label_index: int, label_name: str, confidence: float).
    """
    padded      = preprocess(text, tokenizer)
    probs       = model.predict(padded, verbose=0)[0]   # shape (3,)
    label_idx   = int(np.argmax(probs))
    confidence  = float(probs[label_idx])

    # Decode nhãn
    if isinstance(label_encoder, dict):
        # label_encoder lưu dạng {tên: idx} → cần đảo ngược
        idx_to_name = {v: k for k, v in label_encoder.items()}
        raw_name = idx_to_name.get(label_idx, f"Class {label_idx}")
    elif hasattr(label_encoder, "inverse_transform"):
        raw_name = str(label_encoder.inverse_transform([label_idx])[0])
    else:
        raw_name = str(label_idx)

    # Ưu tiên LABEL_MAP hiển thị đẹp, fallback về raw_name
    label_name = LABEL_MAP.get(label_idx, raw_name)

    return label_idx, label_name, confidence


# ─────────────────────────────────────────────────────────────────────────────
# GIAO DIỆN TKINTER
# ─────────────────────────────────────────────────────────────────────────────
class SentimentApp:
    """
    Lớp chính bao gồm toàn bộ giao diện Tkinter và logic điều phối.
    model, tokenizer, label_encoder được lưu như thuộc tính instance
    (không dùng global) và chỉ load 1 lần duy nhất khi khởi động.
    """

    def __init__(self, root: tk.Tk):
        self.root          = root
        self.model         = None
        self.tokenizer     = None
        self.label_encoder = None
        self._is_loading   = True

        self._setup_window()
        self._build_ui()
        # Load model trong thread riêng để không đóng băng UI
        threading.Thread(target=self._load_in_background, daemon=True).start()

    # ── Cấu hình cửa sổ ───────────────────────────────────────────────────
    def _setup_window(self):
        self.root.title("Phân Loại Cảm Xúc Văn Bản — Nhóm 2")
        self.root.geometry("760x580")
        self.root.minsize(620, 500)
        self.root.configure(bg=COLORS["bg_dark"])
        # Icon emoji trên taskbar (Windows hỗ trợ)
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass
        # Căn giữa màn hình
        self.root.update_idletasks()
        w, h = 760, 580
        x = (self.root.winfo_screenwidth()  - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ── Build UI ──────────────────────────────────────────────────────────
    def build_ui(self):
        """Alias public để build_ui có thể được gọi từ ngoài (tham chiếu theo tên hàm)."""
        self._build_ui()

    def _build_ui(self):
        # ─ Font ─
        try:
            title_font  = tkfont.Font(family="Segoe UI", size=18, weight="bold")
            label_font  = tkfont.Font(family="Segoe UI", size=10)
            input_font  = tkfont.Font(family="Segoe UI", size=12)
            result_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
            hint_font   = tkfont.Font(family="Segoe UI", size=9, slant="italic")
            status_font = tkfont.Font(family="Segoe UI", size=9)
        except Exception:
            title_font = result_font = label_font = input_font = hint_font = status_font = None

        root = self.root

        # ══ HEADER ═══════════════════════════════════════════════════════
        header = tk.Frame(root, bg=COLORS["bg_card"], padx=24, pady=18)
        header.pack(fill="x", side="top")

        tk.Label(
            header,
            text="🧠  Phân Loại Cảm Xúc Văn Bản",
            font=title_font,
            fg=COLORS["accent"],
            bg=COLORS["bg_card"],
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Mô hình Bidirectional LSTM • 3 nhãn: Tích cực / Tiêu cực / Trung lập",
            font=hint_font,
            fg=COLORS["text_muted"],
            bg=COLORS["bg_card"],
        ).pack(anchor="w", pady=(2, 0))

        # Đường kẻ phân cách
        tk.Frame(root, bg=COLORS["border"], height=1).pack(fill="x")

        # ══ BODY ══════════════════════════════════════════════════════════
        body = tk.Frame(root, bg=COLORS["bg_dark"], padx=28, pady=20)
        body.pack(fill="both", expand=True)

        # ─ Nhãn ô nhập ─
        tk.Label(
            body,
            text="✏️  Nhập văn bản cần phân loại:",
            font=label_font,
            fg=COLORS["text_main"],
            bg=COLORS["bg_dark"],
        ).pack(anchor="w", pady=(0, 6))

        # ─ Khung input ─
        input_frame = tk.Frame(body, bg=COLORS["border"], pady=1, padx=1)
        input_frame.pack(fill="x")

        self.text_input = tk.Text(
            input_frame,
            height=5,
            font=input_font,
            bg=COLORS["bg_input"],
            fg=COLORS["text_main"],
            insertbackground=COLORS["accent"],
            relief="flat",
            padx=12,
            pady=10,
            wrap="word",
            undo=True,
        )
        self.text_input.pack(fill="x")
        self.text_input.bind("<Control-Return>", lambda e: self._on_classify())

        # Placeholder
        self._placeholder = "Ví dụ: Hôm nay tôi rất vui vì được điểm cao!"
        self._show_placeholder()
        self.text_input.bind("<FocusIn>",  self._on_focus_in)
        self.text_input.bind("<FocusOut>", self._on_focus_out)

        # ─ Nút hành động ─
        btn_frame = tk.Frame(body, bg=COLORS["bg_dark"])
        btn_frame.pack(fill="x", pady=(14, 0))

        self.btn_classify = tk.Button(
            btn_frame,
            text="🔍  Phân loại",
            font=label_font,
            bg=COLORS["accent"],
            fg="white",
            activebackground=COLORS["btn_hover"],
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._on_classify,
        )
        self.btn_classify.pack(side="left", padx=(0, 10))

        self.btn_clear = tk.Button(
            btn_frame,
            text="🗑️  Xóa",
            font=label_font,
            bg=COLORS["bg_input"],
            fg=COLORS["text_muted"],
            activebackground=COLORS["border"],
            activeforeground=COLORS["text_main"],
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._on_clear,
        )
        self.btn_clear.pack(side="left")

        # Gợi ý phím tắt
        tk.Label(
            btn_frame,
            text="Ctrl+Enter để phân loại",
            font=hint_font,
            fg=COLORS["text_muted"],
            bg=COLORS["bg_dark"],
        ).pack(side="right", padx=4)

        # ─ Phân cách ─
        tk.Frame(body, bg=COLORS["border"], height=1).pack(fill="x", pady=(18, 0))

        # ─ Vùng kết quả ─
        result_outer = tk.Frame(body, bg=COLORS["bg_card"], padx=18, pady=14)
        result_outer.pack(fill="x", pady=(14, 0))

        tk.Label(
            result_outer,
            text="📊  Kết quả phân loại",
            font=label_font,
            fg=COLORS["text_muted"],
            bg=COLORS["bg_card"],
        ).pack(anchor="w", pady=(0, 8))

        self.result_var = tk.StringVar(value="—")
        self.result_label = tk.Label(
            result_outer,
            textvariable=self.result_var,
            font=result_font,
            fg=COLORS["text_main"],
            bg=COLORS["bg_card"],
            wraplength=640,
            justify="left",
        )
        self.result_label.pack(anchor="w")

        # ─ Thanh trạng thái ─
        status_bar = tk.Frame(root, bg=COLORS["bg_card"], padx=16, pady=5)
        status_bar.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(value="⏳  Đang tải model và tokenizer…")
        tk.Label(
            status_bar,
            textvariable=self.status_var,
            font=status_font,
            fg=COLORS["text_muted"],
            bg=COLORS["bg_card"],
        ).pack(side="left")

    # ── Placeholder helpers ───────────────────────────────────────────────
    def _show_placeholder(self):
        self.text_input.insert("1.0", self._placeholder)
        self.text_input.config(fg=COLORS["text_muted"])
        self._has_placeholder = True

    def _on_focus_in(self, event):
        if getattr(self, "_has_placeholder", False):
            self.text_input.delete("1.0", "end")
            self.text_input.config(fg=COLORS["text_main"])
            self._has_placeholder = False

    def _on_focus_out(self, event):
        content = self.text_input.get("1.0", "end").strip()
        if not content:
            self._show_placeholder()

    # ── Load model trong background thread ───────────────────────────────
    def _load_in_background(self):
        try:
            model, tokenizer, label_encoder = load_resources()
            self.model         = model
            self.tokenizer     = tokenizer
            self.label_encoder = label_encoder
            self._is_loading   = False
            self.root.after(0, self._on_load_success)
        except FileNotFoundError as exc:
            self.root.after(0, lambda: self._on_load_error(str(exc)))
        except Exception as exc:
            self.root.after(0, lambda: self._on_load_error(
                f"Lỗi không xác định khi tải model:\n{exc}"
            ))

    def _on_load_success(self):
        self.status_var.set("✅  Model sẵn sàng — Nhập văn bản và nhấn Phân loại")
        self.btn_classify.config(state="normal")

    def _on_load_error(self, msg: str):
        self.status_var.set("❌  Lỗi tải model — Xem thông báo")
        messagebox.showerror("Lỗi tải model", msg, parent=self.root)

    # ── Xử lý nút Phân loại ──────────────────────────────────────────────
    def _on_classify(self, event=None):
        if self._is_loading:
            messagebox.showinfo("Vui lòng chờ", "Model đang được tải, hãy thử lại sau giây lát.", parent=self.root)
            return
        if self.model is None:
            messagebox.showerror("Lỗi", "Model chưa được tải. Khởi động lại ứng dụng.", parent=self.root)
            return

        # Đọc và kiểm tra input
        raw = self.text_input.get("1.0", "end").strip()
        if not raw or raw == self._placeholder.strip():
            messagebox.showwarning("Input trống", "Vui lòng nhập văn bản trước khi phân loại.", parent=self.root)
            return

        # Cập nhật trạng thái và chạy inference
        self.status_var.set("⏳  Đang phân tích…")
        self.btn_classify.config(state="disabled")
        self.result_var.set("…")
        self.root.update_idletasks()

        try:
            label_idx, label_name, confidence = predict(
                raw, self.model, self.tokenizer, self.label_encoder
            )
            # Hiển thị kết quả
            display = f"Nhãn: {label_name}   |   Độ tin cậy: {confidence * 100:.1f}%"
            self.result_var.set(display)
            self.result_label.config(fg=LABEL_COLOR.get(label_idx, COLORS["text_main"]))
            self.status_var.set("✅  Phân loại hoàn tất")
        except Exception as exc:
            self.result_var.set("❌  Phân loại thất bại")
            self.result_label.config(fg=COLORS["negative"])
            messagebox.showerror("Lỗi inference", str(exc), parent=self.root)
            self.status_var.set("❌  Lỗi trong quá trình phân loại")
        finally:
            self.btn_classify.config(state="normal")

    # ── Xử lý nút Xóa ────────────────────────────────────────────────────
    def _on_clear(self):
        self.text_input.delete("1.0", "end")
        self._show_placeholder()
        self.result_var.set("—")
        self.result_label.config(fg=COLORS["text_main"])
        self.status_var.set("✅  Model sẵn sàng — Nhập văn bản và nhấn Phân loại"
                            if not self._is_loading else "⏳  Đang tải model và tokenizer…")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = SentimentApp(root)
    root.mainloop()
