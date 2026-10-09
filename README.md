# SRO-Edit 核心代码

## 内容

- `sro_edit/models/roi_head.py`：超体素 ROI 评分头。
- `sro_edit/models/refine_head.py`：区域条件残差 refiner；只在训练阶段使用。
- `sro_edit/roi.py`：软前景占比标签、阈值 ROI 与点击超体素并集、半径 2 的三维球形点击编码、ROI 状态门控。
- `sro_edit/losses.py`：Stage A ROI 损失、Stage B1 编辑/保持损失、Stage B2 ROI 内 teacher consistency 损失。
- `sro_edit/paper_config.py`：论文报告的模型、训练参数和数据集 ROI 阈值。

## 方法流程

输入由外部预计算的 3DSEEDS 超体素标签提供（每例最多 50,000 个）。ROI 头接收图像及累积正、负点击图，对超体素内特征做均值池化并输出相关性分数。Stage A 用超体素前景占比作软标签；推理时按论文阈值选取候选超体素，并并入所有含点击的超体素。Stage B1 学习残差修正，Stage B2 用冻结 refiner 在 ROI 内产生软目标。推理时移除 refiner，并在 ROI 内采用当前二值预测、ROI 外保留上一轮状态。

体数据数组顺序为 `(D,H,W)`；网络张量顺序为 `(B,C,D,H,W)`。推理状态初始化为全零，原始分割 logits 按 0.5 阈值转成二值预测后再做 ROI 门控。`make_refiner_input` 按论文顺序拼接 `[X, I_k, M_edit,k, sigmoid(B_k)]`。调用方负责提供 3D U-Net、交互状态和预计算超体素。

## 参数

超体素上限为 50,000/例；点击图为半径 2 voxel 的二值三维球。Stage A 的特征通道/卷积块数为 32/3，评分 MLP 为 `32–64–1`；ROI 损失参数为 `lambda_size=0.10`、`lambda_recall=1.00`、`alpha=0.25`、`gamma=2.0`。B1 refiner 为 4 个 32 通道残差块，`lambda_edit=0.10`；B2 为 4 轮，`lambda_ft=0.20`。ROI 阈值：BraTS21 WT/TC/ET 为 `0.10/0.12/0.14`；BraTS19 WT/TC/ET 为 `0.22/0.30/0.23`；UCSF-BMSR 为 `0.41`；CFB-GBM 为 `0.64`。

---

# SRO-Edit Core Code

## Contents

- `sro_edit/models/roi_head.py`: supervoxel ROI scoring head.
- `sro_edit/models/refine_head.py`: ROI-conditioned residual refiner used during training.
- `sro_edit/roi.py`: soft foreground-occupancy targets, thresholded ROI with clicked-supervoxel union, radius-2 3D click spheres, and ROI-gated state updates.
- `sro_edit/losses.py`: Stage A ROI loss, Stage B1 edit/preservation loss, and Stage B2 teacher-consistency loss within the ROI.
- `sro_edit/paper_config.py`: reported model and training settings, plus dataset-specific ROI thresholds.

## Method Workflow

The input includes precomputed 3DSEEDS supervoxel labels, with at most 50,000 supervoxels per case. The ROI head takes the image and accumulated positive and negative click maps, mean-pools features within each supervoxel, and predicts relevance scores. Stage A uses supervoxel foreground occupancy as a soft target. At inference, supervoxels above the paper's threshold are selected and combined with all click-containing supervoxels. Stage B1 learns residual corrections, while Stage B2 uses the frozen refiner to produce soft targets within the ROI. The refiner is removed at inference; the current binary prediction is used inside the ROI, and the previous state is retained outside it.

Volumes use array order `(D,H,W)` and network tensors use `(B,C,D,H,W)`. The inference state starts at zero. Raw segmentation logits are thresholded at 0.5 to obtain a binary prediction before ROI gating. `make_refiner_input` concatenates `[X, I_k, M_edit,k, sigmoid(B_k)]` in the order specified by the paper. The caller provides the 3D U-Net, interaction state, and precomputed supervoxels.

## Parameters

The supervoxel limit is 50,000 per case, and click maps are binary 3D spheres with a radius of 2 voxels. Stage A uses 32 feature channels across 3 convolutional blocks and a `32–64–1` scoring MLP. ROI-loss parameters are `lambda_size=0.10`, `lambda_recall=1.00`, `alpha=0.25`, and `gamma=2.0`. The B1 refiner has four residual blocks with 32 channels and `lambda_edit=0.10`; B2 uses four rounds and `lambda_ft=0.20`. ROI thresholds are BraTS21 WT/TC/ET `0.10/0.12/0.14`; BraTS19 WT/TC/ET `0.22/0.30/0.23`; UCSF-BMSR `0.41`; and CFB-GBM `0.64`.
