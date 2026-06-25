# Li DeepLabV3+ Segmentation

这是一个基于 PyTorch 的 DeepLabV3+ 语义分割工程，默认用于 4 类矿物图像分割：

- `background`
- `Feldspar`
- `Quartz`
- `Lepidolite`

项目支持 MobileNetV2 / Xception backbone，包含训练、批量预测、mIoU 评估回调和 ONNX 导出工具。

## 项目结构

```text
.
├── train.py                  # 训练入口
├── predict.py                # 批量预测入口
├── deeplab.py                # 推理封装、ONNX 导出、mIoU mask 生成
├── nets/
│   ├── deeplabv3_plus.py     # DeepLabV3+ 网络结构
│   ├── deeplabv3_training.py # 损失函数、学习率调度、初始化
│   ├── mobilenetv2.py        # MobileNetV2 backbone
│   └── xception.py           # Xception backbone
├── utils/
│   ├── dataloader.py         # VOC 数据集读取与数据增强
│   ├── utils_fit.py          # 单 epoch 训练/验证/保存
│   ├── utils_metrics.py      # mIoU、mPA、Precision 等指标
│   ├── callbacks.py          # loss 与 mIoU 记录回调
│   └── utils.py              # 通用工具函数
└── model_data/               # 预训练权重或自定义权重存放目录
```

## 环境要求

建议使用 Python 3.9+。主要依赖如下：

```bash
pip install torch torchvision numpy opencv-python pillow tqdm matplotlib scipy tensorboard
```

如果需要导出 ONNX：

```bash
pip install onnx onnxsim
```

请根据本机 CUDA 版本安装匹配的 PyTorch。

## 数据集格式

训练数据按 VOC 目录组织：

```text
VOCdevkit/
└── VOC2007/
    ├── JPEGImages/
    │   ├── image_001.jpg
    │   └── image_002.jpg
    ├── SegmentationClass/
    │   ├── image_001.png
    │   └── image_002.png
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

说明：

- `JPEGImages` 存放原图，默认读取 `.jpg`。
- `SegmentationClass` 存放单通道标签图，像素值表示类别编号。
- `train.txt` 和 `val.txt` 每行写一个不带扩展名的图片 id。
- 标签值应从 `0` 开始，且 `num_classes` 需要包含背景类。
- 大于等于 `num_classes` 的标签值会作为 ignore 区域处理。

## 训练

先在 `train.py` 顶部配置以下关键参数：

```python
num_classes = 4
backbone = "mobilenet"
model_path = "model_data/deeplab_mobilenetv2.pth"
VOCdevkit_path = "VOCdevkit"
input_shape = [512, 512]
```

然后运行：

```bash
python train.py
```

训练输出默认保存到 `logs/`：

- `best_epoch_weights.pth`：验证集 loss 最优权重
- `last_epoch_weights.pth`：最后一轮权重
- `epoch_loss.txt` / `epoch_val_loss.txt`：loss 记录
- `epoch_loss.png`：loss 曲线
- `epoch_miou.txt` / `epoch_miou.png`：mIoU 记录

如果 `model_path` 指向的文件不存在，脚本会从头训练并打印提示。

## 批量预测

默认读取 `val/`，输出到 `img_out/`：

```bash
python predict.py
```

指定权重、输入和输出目录：

```bash
python predict.py --model-path logs/best_epoch_weights.pth --input val --output img_out
```

常用参数：

```bash
python predict.py --no-cuda
python predict.py --mix-type 1
python predict.py --count
```

`mix_type` 含义：

- `0`：预测 mask 与原图混合
- `1`：只保存彩色 mask
- `2`：只保留非背景区域

## 类别与颜色

默认类别在 `predict.py` 的 `DEFAULT_CLASSES` 中定义；训练类别数在 `train.py` 的 `num_classes` 中定义。修改类别时，应同时确保：

- 标签图像素值与类别顺序一致。
- `num_classes` 等于类别总数，包含背景。
- 推理时的类别名称列表与训练保持一致。

## 常见问题

### 找不到权重

推理默认读取 `logs/best_epoch_weights.pth`。如果权重保存在其他位置，请使用：

```bash
python predict.py --model-path path/to/weights.pth
```

### 找不到 train.txt 或 val.txt

请检查数据集是否放在：

```text
VOCdevkit/VOC2007/ImageSets/Segmentation/
```

### 显存不足

优先调小：

- `Freeze_batch_size`
- `Unfreeze_batch_size`
- `input_shape`

### 类别数不匹配

如果修改了类别数，旧的分类头权重形状可能不匹配。训练脚本会只加载形状一致的参数，其余参数重新初始化。
