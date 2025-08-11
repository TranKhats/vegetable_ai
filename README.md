# Vegetable AI Classification Project

Dự án phân loại rau củ siêu thị sử dụng YOLOv8 với preprocessing pipeline hoàn chỉnh.

trong siêu thị, khách hàng mua rau củ quả, bỏ vào bịch (mỗi bịch quy định 1 sản phẩm), và nhân viên cân sau đó dán nhãn định giá. Nhân viên phải thuộc các mã rau củ quả. Có thể áp dụng trí tuệ nhân tạo để tự động khi bỏ lên cân, hệ thống sẽ phát hiện loại rau củ quả và xuất thành tiền. Thay vì nhân viên nhớ mã thủ công và nhập.
Vì Bỏ trong bao mờ, nên tôi cần finetune với tập đataset (ngày mai tôi sẽ đi siêu thị chụp ảnh)
Bây giờ tôi cần gán nhãn với label trước được không? Tôi chưa hiểu gán nhãn là gì

## Setup LabelImg with Python 3.9 Virtual Environment

```bash
# Install Python 3.9 if not available
pyenv install 3.9.18

# Create virtual environment with Python 3.9
~/.pyenv/versions/3.9.18/bin/python -m venv labeling_env

# Activate virtual environment
source labeling_env/bin/activate

# Install labelImg in the virtual environment
pip install labelImg 

# Run labelImg (make sure virtual environment is activated)
labelImg data/raw data/classes/classes.txt data/labels

# To deactivate virtual environment when done
# deactivate
```



## 🌟 Tính năng

- **Preprocessing Pipeline**: Xử lý ảnh đầu vào với enhancement, denoising, resizing
- **Data Augmentation**: Tăng cường dữ liệu training với các biến đổi ngẫu nhiên
- **YOLOv8 Training**: Fine-tune YOLOv8 cho bài toán phân loại rau củ
- **Real-time Prediction**: Chạy inference trên ảnh, video hoặc batch images
- **Visualization**: Trực quan hóa kết quả training và predictions
- **Export Models**: Xuất model sang nhiều format khác nhau

## 📁 Cấu trúc thư mục

```
vegetable_ai/
│
├── data/
│   ├── raw/                  # Ảnh chụp gốc (chưa xử lý)
│   ├── processed/            # Ảnh sau preprocessing
│   ├── labels/               # Nhãn YOLO (txt)
│   └── dataset.yaml          # File cấu hình YOLOv8
│
├── preprocessing/
│   ├── __init__.py
│   └── pipeline.py           # Xử lý ảnh đầu vào
│
├── yolo/
│   ├── train.py              # Fine-tune YOLOv8
│   └── predict.py            # Chạy inference
│
├── requirements.txt          # Thư viện cần thiết
├── main.py                   # Chạy pipeline + YOLO
└── README.md                 # File này
```

## 🚀 Cài đặt

### 1. Clone repository và cài đặt dependencies

```bash
cd vegetable_ai
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt ảnh rau củ của bạn vào thư mục `data/raw/`. Bạn có thể tổ chức theo cấu trúc:

```
data/raw/
├── carrot/
│   ├── carrot_001.jpg
│   ├── carrot_002.jpg
│   └── ...
├── potato/
│   ├── potato_001.jpg
│   └── ...
└── ...
```

Hoặc đặt tất cả ảnh vào `data/raw/` và gán nhãn sau.

## 📊 Các loại rau củ được hỗ trợ

Hiện tại model hỗ trợ 10 loại rau củ:
1. **Carrot** (Cà rốt)
2. **Potato** (Khoai tây)
3. **Tomato** (Cà chua)
4. **Cucumber** (Dưa chuột)
5. **Onion** (Hành tây)
6. **Cabbage** (Bắp cải)
7. **Lettuce** (Xà lách)
8. **Bell Pepper** (Ớt chuông)
9. **Broccoli** (Súp lơ xanh)
10. **Eggplant** (Cà tím)

## 🛠️ Sử dụng

### Chế độ tương tác (Recommended)

```bash
python main.py
```

Chương trình sẽ tự động kiểm tra trạng thái và gợi ý bước tiếp theo.

### Chế độ command line

#### 1. Preprocessing dữ liệu

```bash
python main.py --mode preprocess
```

#### 2. Training model

```bash
python main.py --mode train
```

#### 3. Chạy prediction

```bash
# Predict single image
python main.py --mode predict --source path/to/image.jpg

# Predict batch images
python main.py --mode predict --source path/to/image/directory

# Predict video
python main.py --mode predict --source path/to/video.mp4
```

#### 4. Chạy full pipeline

```bash
python main.py --mode full --raw-data path/to/raw/images
```

#### 5. Đánh giá model

```bash
python main.py --mode evaluate
```

#### 6. Export model

```bash
python main.py --mode export --export-format onnx
```

### Sử dụng các module riêng lẻ

#### Preprocessing

```python
from preprocessing.pipeline import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(640, 640))
preprocessor.process_dataset("data/raw", "data/processed")
preprocessor.augment_dataset("data/processed")
```

#### Training

```python
from yolo.train import VegetableYOLOTrainer

trainer = VegetableYOLOTrainer('yolov8n.pt', 'data/dataset.yaml')
results = trainer.train(epochs=100, batch_size=16)
```

#### Prediction

```python
from yolo.predict import VegetablePredictor

predictor = VegetablePredictor('path/to/best.pt')
predictions = predictor.predict_image('path/to/image.jpg')
```

## ⚙️ Configuration

Bạn có thể tùy chỉnh cấu hình trong `main.py` hoặc tạo file config JSON:

```json
{
  "data": {
    "target_size": [640, 640],
    "split_ratios": [0.7, 0.2, 0.1],
    "augment_factor": 3
  },
  "training": {
    "model_name": "yolov8n.pt",
    "epochs": 100,
    "batch_size": 16,
    "lr0": 0.01
  },
  "prediction": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }
}
```

Sử dụng: `python main.py --mode full --config config.json`

## 📈 Kết quả Training

Sau khi training, bạn sẽ tìm thấy:

- **Model weights**: `runs/detect/experiment_name/weights/best.pt`
- **Training plots**: `runs/detect/experiment_name/training_plots.png`
- **Validation results**: `runs/detect/experiment_name/val_batch*_pred.jpg`
- **Training logs**: `runs/detect/experiment_name/results.csv`

## 🔍 Annotation Tools

Để tạo labels cho YOLO, bạn có thể sử dụng:

1. **LabelImg**: https://github.com/tzutalin/labelImg
2. **CVAT**: https://github.com/openvinotoolkit/cvat
3. **Roboflow**: https://roboflow.com/
4. **Label Studio**: https://labelstud.io/

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pillow
- Albumentations
- Matplotlib
- Pandas

## 🎯 Performance Tips

1. **GPU Training**: Sử dụng GPU để tăng tốc training
2. **Batch Size**: Điều chỉnh batch size phù hợp với VRAM
3. **Image Size**: 640x640 là optimal cho YOLOv8
4. **Augmentation**: Tăng augment_factor nếu có ít data
5. **Pretrained Weights**: Sử dụng pretrained weights để converge nhanh hơn

## 🐛 Troubleshooting

### Lỗi CUDA out of memory
```bash
# Giảm batch size
python main.py --mode train  # Default batch_size=16
```

### Không tìm thấy images
- Kiểm tra format ảnh (.jpg, .jpeg, .png)
- Đảm bảo ảnh không bị corrupt

### Model không converge
- Kiểm tra quality dữ liệu
- Tăng số epochs
- Điều chỉnh learning rate

## 📄 License

Dự án này sử dụng các thư viện open source. Vui lòng kiểm tra license của từng thành phần.

## 🤝 Contributing

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📞 Support

Nếu bạn gặp vấn đề, vui lòng:

1. Kiểm tra phần Troubleshooting
2. Xem logs trong file `vegetable_ai.log`
3. Tạo issue với thông tin chi tiết

---

**Happy Coding! 🥕🥔🍅**
