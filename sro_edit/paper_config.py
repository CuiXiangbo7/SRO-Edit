"""Reported SRO-Edit settings from the supplied manuscript."""

MAX_SUPERVOXELS_PER_CASE = 50_000
CLICK_SPHERE_RADIUS_VOXELS = 2

ROI_HEAD = {
    "dimensions": 3,
    "feature_channels": 32,
    "convolution_blocks": 3,
    "scoring_mlp": (32, 64, 1),
}

ROI_LOSS = {
    "lambda_size": 0.10,
    "lambda_recall": 1.00,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
}

STAGE_A = {
    "epochs": 100,
    "optimizer": "AdamW",
    "batch_size": 1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patch_size_dhw": (64, 128, 128),
}

STAGE_B1 = {
    "residual_blocks": 4,
    "width": 32,
    "lambda_edit": 0.10,
    "epochs": 100,
    "optimizer": "AdamW",
    "batch_size": 1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patch_size_dhw": (64, 128, 128),
}

STAGE_B2 = {
    "rounds": 4,
    "epochs": 100,
    "optimizer": "SGD",
    "learning_rate": 1e-2,
    "weight_decay": 3e-5,
    "lambda_ft": 0.20,
}

INITIAL_BACKBONE = {
    "epochs": 250,
    "optimizer": "SGD",
    "batch_size": 2,
    "learning_rate": 1e-2,
    "weight_decay": 3e-5,
}

ROI_THRESHOLDS = {
    "BraTS21": {"WT": 0.10, "TC": 0.12, "ET": 0.14},
    "BraTS19": {"WT": 0.22, "TC": 0.30, "ET": 0.23},
    "UCSF-BMSR": {"enhancing_tumor": 0.41},
    "CFB-GBM": {"GTV": 0.64},
}

PAPER_PYTORCH_VERSION = "2.7.1"
