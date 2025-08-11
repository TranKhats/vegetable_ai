# Vegetable Video Extraction Guide

## 📁 Cấu trúc thư mục

```
video_extraction/
├── extract_vegetable_frames.py    # Script chính
├── EXTRACT_VEGETABLE_GUIDE.md    # Hướng dẫn này
└── videos/
    ├── carot/          # Thư mục cà rốt
    │   ├── IMG_6871.MOV
    │   └── IMG_6872.MOV
    ├── tomato/         # Thư mục cà chua  
    │   ├── video1.mp4
    │   └── video2.mp4
    └── potato/         # Thư mục khoai tây
        └── video3.mov
```

## 🚀 Cách sử dụng

### 1. Extract tất cả rau củ

```bash
# Chuyển vào thư mục video_extraction
cd video_extraction

# Extract từ tất cả thư mục rau củ
python extract_vegetable_frames.py

# Kết quả sẽ được lưu vào:
# ../data/raw/carot/carot_IMG_6871_0001_t2.0s.jpg
# ../data/raw/carot/carot_IMG_6872_0001_t2.0s.jpg
# ../data/raw/tomato/tomato_video1_0001_t2.0s.jpg
# ../data/raw/potato/potato_video3_0001_t2.0s.jpg
```

### 2. Extract chỉ 1 loại rau củ

```bash
# Trong thư mục video_extraction
# Chỉ extract cà rốt
python extract_vegetable_frames.py --vegetable carot

# Chỉ extract cà chua
python extract_vegetable_frames.py --vegetable tomato
```

### 3. Điều chỉnh tần suất extract

```bash
# Extract ít hơn (mỗi 120 frames ~ 4 giây)
python extract_vegetable_frames.py --interval 120

# Extract nhiều hơn (mỗi 30 frames ~ 1 giây)  
python extract_vegetable_frames.py --interval 30

# Extract từ cà rốt với interval 90 frames
python extract_vegetable_frames.py --vegetable carot --interval 90
```

### 4. Liệt kê các loại rau củ có sẵn

```bash
python extract_vegetable_frames.py --list
```

## 📊 Ước tính kết quả

| Video Length | Interval | Images per Video |
|-------------|----------|------------------|
| 1 phút      | 60 frames| ~30 ảnh         |
| 2 phút      | 60 frames| ~60 ảnh         |
| 1 phút      | 30 frames| ~60 ảnh         |
| 5 phút      | 120 frames| ~75 ảnh        |

## 🎯 Kết quả mong đợi

Sau khi chạy, bạn sẽ có:

```
data/raw/    # Ở thư mục gốc (không phải trong video_extraction)
├── carot/
│   ├── carot_IMG_6871_0001_t2.0s.jpg
│   ├── carot_IMG_6871_0002_t4.0s.jpg
│   ├── carot_IMG_6872_0001_t2.0s.jpg
│   └── carot_IMG_6872_0002_t4.0s.jpg
├── tomato/
│   ├── tomato_video1_0001_t2.0s.jpg
│   └── tomato_video1_0002_t4.0s.jpg
└── potato/
    └── potato_video3_0001_t2.0s.jpg
```

## 💡 Workflow đề xuất

1. **Tạo thư mục cho từng loại rau củ:**
   ```bash
   mkdir -p videos/{carot,tomato,potato,cabbage,onion}
   ```

2. **Copy video vào thư mục tương ứng**

3. **Extract frames:**
   ```bash
   cd video_extraction
   python extract_vegetable_frames.py
   ```

4. **Kiểm tra kết quả:**
   ```bash
   ls -la ../data/raw/*/
   ```

5. **Tiếp tục với labelImg để gán nhãn**

## ⚙️ Cài đặt mặc định

- **Frame interval:** 60 frames (≈ 2 giây với video 30fps)
- **JPEG quality:** 95%
- **Format:** `.jpg`
- **Naming:** `{vegetable}_{video_name}_{number}_t{timestamp}s.jpg`

## 🔧 Tuỳ chỉnh interval theo nhu cầu

- **Interval = 30:** Nhiều ảnh, tốt cho object detection
- **Interval = 60:** Cân bằng, đủ dữ liệu
- **Interval = 120:** Ít ảnh, tránh trùng lặp
- **Interval = 240:** Rất ít ảnh, chỉ những frame quan trọng


tôi muốn viết lại từ đầu quá trình extract_frame đáp ứng các tiêu chí sau:

video source tôi để ở video_extraction/videos/{name vegatables} sau khi run video_extraction tôi muốn được các ảnh có trong data/raw/{name vegetable}
Target size: 640x640 (YOLOv8 optimal)
Chuyển đổi sang không gian màu phù hợp (RGB hoặc HSV).
Dùng bộ lọc Gaussian hoặc Median để làm mịn ảnh.
Augmentation:
5.1. Xoay ảnh
5.2. Lật ngang/dọc
5.3. Zoom/crop
5.4. Thay đổi ánh sáng