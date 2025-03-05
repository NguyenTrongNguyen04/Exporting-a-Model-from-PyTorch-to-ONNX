# ONNX Runtime - Chạy mô hình trên ảnh bằng ONNX Runtime

## Giới thiệu

ONNX Runtime là một công cụ giúp chạy mô hình AI trên nhiều nền tảng khác nhau, tối ưu hóa hiệu suất trên cả CPU và GPU. Trong hướng dẫn này, chúng ta sẽ thực hiện các bước sau:

1. **Xuất mô hình từ PyTorch sang ONNX**
2. **Nạp mô hình và chạy thử với dữ liệu đầu vào giả lập**
3. **Chạy mô hình trên ảnh thực tế và so sánh kết quả**

## Cài đặt

Trước tiên, cần cài đặt các thư viện cần thiết:

```bash
pip install onnxruntime torchvision pillow numpy
```

## Hướng dẫn sử dụng

### 1. Chạy mô hình với dữ liệu giả lập

```python
import onnxruntime as ort
import torch
import numpy as np

# Load mô hình ONNX
ort_session = ort.InferenceSession("model.onnx")

# Tạo dữ liệu đầu vào giả lập
dummy_input = torch.randn(1, 3, 224, 224).numpy()

# Chạy mô hình
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
ort_outs = ort_session.run(None, ort_inputs)

print("Output shape:", ort_outs[0].shape)
```

### 2. Chạy mô hình trên ảnh thực tế

#### Bước 1: Tiền xử lý ảnh

```python
from PIL import Image
import torchvision.transforms as transforms

# Đọc ảnh
img = Image.open("cat.jpg")
resize = transforms.Resize([224, 224])
img = resize(img)

# Chuyển đổi ảnh sang không gian màu YCbCr
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

# Chuyển đổi Y component sang tensor
to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)
```

#### Bước 2: Chạy mô hình ONNX Runtime

```python
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]
```

#### Bước 3: Hậu xử lý và hiển thị ảnh

```python
import matplotlib.pyplot as plt

# Chuyển output thành ảnh
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Hiển thị ảnh
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(final_img)
axes[1].set_title("Super-Resolution Image")
axes[1].axis("off")

plt.show()
```

## Kết luận

- ONNX Runtime giúp chạy mô hình AI trên nhiều nền tảng khác nhau.
- Chúng ta đã thử nghiệm chạy mô hình trên dữ liệu giả lập và ảnh thực tế.
- Mô hình có thể cải thiện chất lượng ảnh bằng phương pháp Super-Resolution.
