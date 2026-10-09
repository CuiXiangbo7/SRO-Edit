# SRO-Edit 核心代码

本目录只保留论文方法的核心模块，参数以随附论文为准。它不是完整训练仓库，也不含数据、checkpoint 或完整实验脚本。

## 内容

- `sro_edit/models/roi_head.py`：超体素 ROI 评分头。
- `sro_edit/models/refine_head.py`：区域条件残差 refiner；只在训练阶段使用。
- `sro_edit/roi.py`：软前景占比标签、阈值 ROI 与点击超体素并集、半径 2 的三维球形点击编码、ROI 状态门控。
- `sro_edit/losses.py`：Stage A ROI 损失、Stage B1 编辑/保持损失、Stage B2 ROI 内 teacher consistency 损失。
- `sro_edit/paper_config.py`：论文报告的模型、训练参数和数据集 ROI 阈值。

## 方法流程

输入由外部预计算的 3DSEEDS 超体素标签提供（每例最多 50,000 个）。ROI 头接收图像及累积正、负点击图，对超体素内特征做均值池化并输出相关性分数。Stage A 用超体素前景占比作软标签；推理时按论文阈值选取候选超体素，并并入所有含点击的超体素。Stage B1 学习残差修正，Stage B2 用冻结 refiner 在 ROI 内产生软目标。推理时移除 refiner，并在 ROI 内采用当前二值预测、ROI 外保留上一轮状态。

体数据数组顺序为 `(D,H,W)`；网络张量顺序为 `(B,C,D,H,W)`。推理状态初始化为全零，原始分割 logits 按 0.5 阈值转成二值预测后再做 ROI 门控。`make_refiner_input` 按论文顺序拼接 `[X, I_k, M_edit,k, sigmoid(B_k)]`。调用方负责提供 3D U-Net、交互状态和预计算超体素。

## 论文参数

超体素上限为 50,000/例；点击图为半径 2 voxel 的二值三维球。Stage A 的特征通道/卷积块数为 32/3，评分 MLP 为 `32–64–1`；ROI 损失参数为 `lambda_size=0.10`、`lambda_recall=1.00`、`alpha=0.25`、`gamma=2.0`。B1 refiner 为 4 个 32 通道残差块，`lambda_edit=0.10`；B2 为 4 轮，`lambda_ft=0.20`。各数据集 ROI 阈值见 `paper_config.py`。

## 范围

未纳入完整 3D U-Net、3DSEEDS 生成器、数据读取/预处理、点击坐标采样器、训练入口、checkpoint、对比基线及消融分支。论文规定从 ROI 内最大的 FN/FP 连通域取中心点击；源码采样器走随机取点路径，因此没有复制该采样器。论文未报告足以固定训练专用 FN 超体素扩展的选择参数，也没有照搬源码中的额外参数。此处实现用于呈现和复用方法核心，不能单独完成论文全流程复现。

`models/roi_head.py` 与 `models/refine_head.py` 从本地 `code/svis/interactive_heads/models/` 选取；其余 ROI 构造、球形点击编码及损失接口按论文描述整理。论文要求半径 2 的三维球，源码 `add_ball` 实际绘制立方区域，所以这里只保留按球形实现的 `encode_click_spheres`。ROI 标签固定使用论文的软前景占比，源码中其他硬标签和额外损失模式没有纳入。源目录保持未修改。本目录附带 SViS 的 Apache-2.0 许可证文本。
