<div align="center">
<br>
<h1>FysicsReason：跨全模态的可验证世界状态推理基准</h1>

<h5 align="center">如果你喜欢我们的项目，欢迎在 GitHub 上给我们一个星标 ⭐ 以获取最新更新。</h5>

<div align="center">
  <a href="https://github.com/Fysics-AI/FysicsReason">🏠 项目主页</a>
  &nbsp;&nbsp;
  <a href="PAPER_LINK">📖 论文</a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Fysics-AI/FysicsReason">🤗 数据集</a>
  &nbsp;&nbsp;
  <a href="README.md">English</a>
</div>

</div>

## 🚀 新闻
- **`2026-09-20`** 我们发布 **FysicsReason**，这是一个面向跨全模态可验证世界状态推理的基准，包含 2,000 多个问答样本，并提供显式的中间世界状态监督。

## 🎯 ***FysicsReason*** 概览
<img src="figs/pic2.png" width="100%" height="100%">

我们提出 ***FysicsReason***，这是一个通过“先感知、再推理”范式评估全模态推理能力的基准，并引入可验证的世界状态变量（World State Variables, WSVs）。该基准将多模态推理拆解为显式的中间状态和最终结果，从而能够更细粒度地评估感知和推理两个环节。

***FysicsReason*** 包含 2,000 个覆盖图像、音频、视频和文本的问答样本，涵盖三类能力、五种 WSV 类型，以及跨五个 WSV 类别的十种细粒度能力。它旨在诊断模型在推理出最终答案之前，是否真正感知到了正确证据。

## 🔍 数据构建
<img src="figs/pic3.png" width="100%" height="100%">

该数据集通过数据收集、问答构建、自动过滤和人工检查构建而成。我们筛选高信息量的公开多模态来源，并将其转换为具备 WSV 感知的推理问题，同时进行相关性和对齐检查。

## 📈 实验结果
<img src="figs/main_table.png" width="100%" height="100%">

我们在文本、音频、图像和视频设置下评估了多个模型。主表总结了各模型在基准任务上的表现，并展示了中间世界状态感知与最终答案正确性之间的差距。

## 🧰 评估

预测文件应为 JSONL 格式，每行一个样本：

```json
{"index":0,"task_source":"task1","wsv_answer":"...","final_answer":"..."}
```

`index` 是 `data/<task_source>/<task_source>.parquet` 中的行索引。
`task_source` 是 `task1`、`task2`、`task3`、`task4`、`task5` 之一。
同一个命令既可以评估单个 task 的 JSONL，也可以评估混合所有 task 的 JSONL。

运行评估：

```bash
/share/apps/miniconda3/envs/download/bin/python \
  eval/judge.py \
  path/to/predictions.jsonl \
  --output path/to/judged.jsonl \
  --summary-output path/to/summary.json
```

对于 task2 的 bbox 评估，请从命令行选择坐标模式：

```bash
--task2-bbox-coordinate-mode auto
--task2-bbox-coordinate-mode absolute
--task2-bbox-coordinate-mode relative_1
--task2-bbox-coordinate-mode relative_1000
```

如果 task2 的 `wsv_answer` bbox 是相对坐标，而不是绝对像素坐标，需要在该 JSONL 行里提供图像尺寸：

```json
{"index":0,"task_source":"task2","wsv_answer":"[0.1,0.2,0.4,0.6]","final_answer":"A","image_width":1280,"image_height":720}
```

`relative_1` 和 `relative_1000` 需要提供 `image_width`、`image_height`；如果使用 `auto` 且模型可能输出相对坐标，也建议提供。使用 `--task2-bbox-coordinate-mode absolute` 且 bbox 为像素坐标时不需要提供尺寸。

输出 JSONL 保留输入字段，并新增 `S_wsv`、`S_r` 和 `S`。summary JSON 按 task 和 `Avg` 报告这些分数。
