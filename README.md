# CFT-AO+A GitHub Package

这个目录是从原始工作仓库中手工筛选出来的一个独立代码包，只保留与 `CFT-AO+A` 相关、且用于训练、评估、复现核心 2D obstacle 实验所需的源码与脚本。

## 目录说明

- 根目录：核心模型、A-wrap、训练脚本和基础依赖模块
- `data_generation/`：与本文主线实验相关的数据生成脚本
- `paper_preparation/`：评估、聚合、成本统计和作图脚本
- `the_third_paper/`：与几何 atlas/OOD 相关的补充评估脚本

## 已包含内容

- `CFT-AO+A` 主干：`cft_ao_2d.py`、`fourier_2d_cft_residual.py`、`chebyshev.py`
- A-wrap：`boundary_ext_residual_2d.py`
- 基础模块：`Adam.py`、`utilities3.py`
- 对比基线：`fourier_2d_baseline.py`、`uno_2d.py`
- 训练脚本：`train_cftaoA_*`、`train_fnoA_*`、`train_unoA_diff2d_obstacle.py`、`train_fno_diff2d_obstacle.py`
- 评估与汇总脚本：`paper_preparation/` 中与 CFT-AO+A 主实验直接相关的脚本
- 数据生成脚本：`data_generation/` 中 obstacle/diffusion/pflow 主线相关脚本

## 刻意未包含

- 论文源文件、投稿材料、PDF、LaTeX 缓存
- 历史备份、无关 3D 实验、其它论文项目
- 数据文件、模型权重、预测结果、中间缓存

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## 运行建议

请从本目录根路径执行脚本，这样现有 `import` 路径可以保持不变。例如：

```bash
python train_cftaoA_diff2d_obstacle.py --help
python paper_preparation/eval_A_models_2d.py --help
```

`paper_preparation/CFTAOA_PRE_SUBMISSION_RUNBOOK.md` 中保留了你之前论文加强实验时使用的主要命令链，可作为复现入口。

## 上传 GitHub 前建议

1. 先只上传这个目录，不要连原始大仓库一起推。
2. 先检查 `git status`，确认没有数据、权重、PDF、`.DS_Store`、`__pycache__` 被带进去。
3. 大文件建议后续单独做 Release 或外链说明，不直接进仓库历史。
