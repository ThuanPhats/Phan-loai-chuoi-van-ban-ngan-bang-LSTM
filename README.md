# Phân Loại Chuỗi Văn Bản Ngắn Bằng LSTM

> 🇬🇧 [English Guide](#english-guide) &nbsp;|&nbsp; 🇻🇳 [Hướng Dẫn Tiếng Việt](#hướng-dẫn-tiếng-việt)

---

## English Guide

### Overview
Short text sentiment classification using a **Bidirectional LSTM** model.  
**3 labels:** Positive / Negative / Neutral

### Requirements
- Python 3.8+
- pip

### Installation & Run
**Option 1 — Auto (Windows)**  
Double-click `run.bat` — installs dependencies and launches the app automatically.

**Option 2 — Manual**
```bash
git clone https://github.com/ThuanPhats/Phan-loai-chuoi-van-ban-ngan-bang-LSTM.git
cd Phan-loai-chuoi-van-ban-ngan-bang-LSTM
pip install -r requirements.txt
python app.py
```

### Project Structure
```
project/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── run.bat                 # Auto-install & launch script (Windows)
├── models/
│   ├── best_model_balanced.h5   # Trained LSTM model
│   ├── tokenizer.pkl            # Fitted tokenizer
│   └── label_encoder.pkl        # Label encoder
└── data/
    ├── raw/                # Original datasets
    └── processed/          # Padded train/val/test splits
```

### Usage
1. Run the app via `run.bat` or `python app.py`
2. Type a short Vietnamese sentence into the input box
3. Click **Phân loại** or press `Ctrl+Enter`
4. The predicted label and confidence score will appear below

---

## Hướng Dẫn Tiếng Việt

### Tổng Quan
Ứng dụng phân loại cảm xúc văn bản ngắn sử dụng mô hình **Bidirectional LSTM**.  
**3 nhãn:** Tích cực / Tiêu cực / Trung lập

### Yêu Cầu
- Python 3.8 trở lên
- pip

### Cài Đặt & Chạy
**Cách 1 — Tự động (Windows)**  
Nhấp đúp vào `run.bat` — tự động cài thư viện và khởi động ứng dụng.

**Cách 2 — Thủ công**
```bash
git clone https://github.com/ThuanPhats/Phan-loai-chuoi-van-ban-ngan-bang-LSTM.git
cd Phan-loai-chuoi-van-ban-ngan-bang-LSTM
pip install -r requirements.txt
python app.py
```

### Cấu Trúc Thư Mục
```
project/
├── app.py                  # File chạy chính
├── requirements.txt        # Danh sách thư viện
├── run.bat                 # Script tự động cài & chạy (Windows)
├── models/
│   ├── best_model_balanced.h5   # Mô hình LSTM đã huấn luyện
│   ├── tokenizer.pkl            # Tokenizer đã fit
│   └── label_encoder.pkl        # Encoder nhãn
└── data/
    ├── raw/                # Dữ liệu gốc
    └── processed/          # Tập train/val/test đã padding
```

### Hướng Dẫn Sử Dụng
1. Chạy app qua `run.bat` hoặc `python app.py`
2. Nhập một câu văn bản tiếng Việt ngắn vào ô nhập liệu
3. Bấm **Phân loại** hoặc nhấn `Ctrl+Enter`
4. Nhãn dự đoán và độ tin cậy sẽ hiển thị bên dưới
