# YOLOv8 Auto-Labelling for Vegetables

Tự động gán nhãn cho các ảnh rau củ chưa được labeled bằng YOLOv8, dựa trên dữ liệu đã được label thủ công.

## 📊 **Current Status**
- ✅ **98 ảnh carrot** đã được label thủ công (trong `yolo_labels/`)
- ❌ **434 ảnh carrot** chưa được label
- 🎯 **Mục tiêu**: Tự động label 434 ảnh còn lại

## 🚀 **Quick Start**

### **1. Cài đặt dependencies**
```bash
cd preparing_datasets/02.auto_labelling

# Cài đặt packages
python run_auto_labelling.py --install-requirements

# Kiểm tra cài đặt
python run_auto_labelling.py --check-requirements
```

### **2. Chạy complete workflow**
```bash
# Auto-label tất cả ảnh carrot còn lại
python run_auto_labelling.py --vegetable carrot

# Hoặc với custom parameters
python run_auto_labelling.py --vegetable carrot --epochs 100 --conf 0.6
```

### **3. Chạy từng bước riêng lẻ**
```bash
# Bước 1: Chuẩn bị dataset
python run_auto_labelling.py --prepare-only --vegetable carrot

# Bước 2: Train model
python run_auto_labelling.py --train-only --vegetable carrot --epochs 50

# Bước 3: Auto-label
python run_auto_labelling.py --label-only --vegetable carrot --conf 0.5
```

## 📁 **File Structure**

```
02.auto_labelling/
├── prepare_dataset.py      # Chuẩn bị dataset cho training
├── train_model.py         # Train YOLOv8 model
├── auto_label.py          # Auto-label ảnh còn lại
├── run_auto_labelling.py  # Workflow chính
├── requirements.txt       # Dependencies
├── README.md             # Documentation này
│
├── dataset_carrot/       # Dataset cho training (được tạo tự động)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── dataset.yaml
│
└── runs/                 # Training results
    └── detect/
        └── carrot_auto_labeller/
            ├── weights/
            │   ├── best.pt      # Model tốt nhất
            │   └── last.pt      # Model cuối cùng
            ├── results.png
            └── ...
```

## 🔧 **Detailed Commands**

### **Dataset Preparation**
```bash
# Chuẩn bị dataset từ labeled data
python prepare_dataset.py --vegetable carrot --train-ratio 0.8

# Kết quả:
# - 78 ảnh cho training (80%)
# - 20 ảnh cho validation (20%)
# - Tạo dataset.yaml config
```


## 🏋️‍♂️ **Model Training Steps**

### **Step 1: Chuẩn bị dataset**
```bash
python prepare_dataset.py --vegetable carrot --train-ratio 0.8
# Tạo thư mục dataset_carrot/ với train/val/images và labels, cùng file dataset.yaml
```

### **Step 2: Training YOLOv8 model**
```bash
# Train với thông số mặc định
python train_model.py --vegetable carrot

# Train với thông số tùy chỉnh
python train_model.py --vegetable carrot --epochs 100 --batch 32 --imgsz 640

# Các tham số phổ biến:
# --epochs: Số vòng lặp huấn luyện (default: 50)
# --batch: Batch size (default: 16)
# --imgsz: Kích thước ảnh (default: 640)
# --weights: Đường dẫn model pretrain (nếu muốn finetune)
# --device: Chọn GPU/CPU (default: auto)
```

### **Step 3: Theo dõi quá trình training**
```bash
# Xem log training, kết quả, và mô hình lưu tại:
runs/detect/carrot_auto_labeller/
# Xem metrics, loss, mAP qua file results.png hoặc TensorBoard:
tensorboard --logdir runs/detect
```

### **Step 4: Đánh giá và sử dụng model**
```bash
# Model tốt nhất sẽ lưu ở:
runs/detect/carrot_auto_labeller/weights/best.pt
# Dùng model này để inference hoặc auto-label tiếp
```


### **Step 5: Training cho tất cả các loại rau củ (multi-class, one-time)**
```bash
# 1. Chuẩn bị dataset tổng hợp (tất cả các loại rau củ)
#copy hết tất cả image của những loại vegetable bỏ vào raw/all 
python prepare_dataset.py --vegetable all --train-ratio 0.8
# 2. Đảm bảo file dataset.yaml chứa đủ đường dẫn images/labels cho tất cả class
# 3. Đảm bảo file classes.txt liệt kê đầy đủ các loại rau củ

# 4. Training YOLOv8 cho toàn bộ dataset chỉ với một lần:
python train_model.py --vegetable all --epochs 100 --batch 32 --imgsz 640
# Model sẽ nhận diện tất cả class trong một lần train, không cần train riêng từng loại
# Kết quả lưu tại runs/detect/all_auto_labeller/
```

### **Auto-Labelling**
```bash
# Auto-label với default threshold
python auto_label.py --vegetable carrot

# Auto-label với custom thresholds
python auto_label.py --vegetable carrot --conf 0.6 --iou 0.5

# Parameters:
# --conf: Confidence threshold (default: 0.5)
# --iou: IoU threshold for NMS (default: 0.45)
# --verify: Verify auto-generated labels
```

## 📊 **Expected Results**

### **Training Performance**
- **mAP50**: 0.85-0.95 (dự kiến)
- **Training time**: 30-60 phút (tùy GPU)
- **Model size**: ~6MB (YOLOv8n)

### **Auto-Labelling Performance**
- **Success rate**: 85-95% ảnh được detect
- **Confidence**: Thường 0.7-0.9 cho carrot rõ ràng
- **Speed**: ~1-2 giây/ảnh

### **Quality Indicators**
- ✅ **High confidence (>0.7)**: Labels có thể tin tưởng
- ⚠️ **Medium confidence (0.5-0.7)**: Cần review thủ công
- ❌ **No detection**: Có thể cần label thủ công

## 🔍 **Quality Control**

### **Verification Process**
```bash
# Verify auto-generated labels
python auto_label.py --vegetable carrot --verify

# Manual review recommended for:
# - Low confidence predictions (<0.7)
# - Images with no detections
# - Unusual augmentation variants
```

### **Post-Processing**
1. **Review low-confidence predictions** manually
2. **Spot-check random samples** for accuracy
3. **Fix obvious errors** in labelImg if needed
4. **Re-train với all labels** nếu cần thiết

## 🎯 **Workflow Optimization**

### **Training Tips**
- **Tăng epochs** (100-200) nếu validation loss vẫn giảm
- **Adjust confidence threshold** based on precision/recall requirements
- **Use larger model** (yolov8s, yolov8m) nếu accuracy chưa đủ

### **Auto-labelling Tips**
- **Lower confidence** (0.3-0.4) để catch more objects, sau đó manual review
- **Higher confidence** (0.7-0.8) để có labels chất lượng cao
- **Batch processing** nhiều vegetables cùng lúc

## 📈 **Progress Tracking**

```bash
# Check current labelling status
python -c "
import os
labeled = len([f for f in os.listdir('../../data/yolo_labels') if f.startswith('carrot')])
total = len([f for f in os.listdir('../../data/raw/carrot') if f.endswith('.jpg')])
print(f'Progress: {labeled}/{total} ({labeled/total*100:.1f}%)')
"
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **CUDA out of memory**
   ```bash
   # Giảm batch size
   python train_model.py --batch 8
   ```

2. **No detections during auto-labelling**
   ```bash
   # Giảm confidence threshold
   python auto_label.py --conf 0.3
   ```

3. **Low training accuracy**
   ```bash
   # Tăng epochs hoặc dùng model lớn hơn
   python train_model.py --epochs 100
   ```

### **Performance Monitoring**
- **Training logs**: `runs/detect/carrot_auto_labeller/`
- **Validation metrics**: Xem results.png
- **TensorBoard**: `tensorboard --logdir runs/detect`

## 🎉 **Success Criteria**

✅ **Complete Success**:
- 95%+ ảnh được auto-label
- mAP50 > 0.85
- <5% labels cần manual correction

⚠️ **Partial Success**:
- 80-95% ảnh được auto-label  
- mAP50 > 0.75
- 5-15% labels cần review

❌ **Needs Improvement**:
- <80% success rate
- mAP50 < 0.75
- >15% incorrect labels

---

## 📞 **Next Steps**

Sau khi hoàn thành auto-labelling cho carrot:

1. **Expand to other vegetables** (tomato, cucumber, etc.)
2. **Multi-class training** với tất cả vegetables
3. **Real-time inference** application
4. **Data augmentation pipeline** optimization
5. **Review by labelImg
cd /Volumes/ktran2/MSE/TuHoc/PhanLoaiRauCuSieuThi/vegetable_ai && labelImg data/raw/carrot
