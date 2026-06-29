# OpenVision：基于 Grounding DINO 的开放词汇目标检测系统

## 1. 项目简介

OpenVision 是一个基于 Grounding DINO 的开放词汇目标检测系统。用户可以上传图片，并输入文本提示词，例如：

```text
person. helmet. hard hat.
cat. remote.
car. bus. traffic light.
```

系统会根据文本提示词在图片中检测相关目标，并输出检测框、类别标签、置信度和 JSON 统计报告。

本项目进一步针对工地安全场景进行了扩展，支持对人员、安全帽、反光背心等目标进行检测，并生成简单的安全风险提示。

---

## 2. 项目功能

当前项目已经实现以下功能：

- 基于文本 prompt 的开放词汇目标检测；
- 支持用户自定义检测类别；
- 输出目标检测框、类别和置信度；
- 生成检测结果可视化图片；
- 生成 JSON 格式检测统计报告；
- 支持通用检测模式和工地安全检测模式；
- 基于检测结果进行简单工地安全风险分析；
- 基于 bbox 空间关系进行逐人风险分析；
- 使用 Gradio 构建交互式网页界面。

---

## 3. 技术路线

项目整体流程如下：

```text
输入图片 + 文本 prompt
↓
Grounding DINO 开放词汇目标检测
↓
检测结果后处理
↓
检测框可视化
↓
统计报告生成
↓
工地安全风险分析
↓
Gradio 页面展示
```

---

## 4. 使用模型

本项目使用 Hugging Face Transformers 版本的 Grounding DINO：

```python
IDEA-Research/grounding-dino-tiny
```

Grounding DINO 是一种开放词汇目标检测模型，可以根据文本提示词定位图片中的目标区域。相比传统 YOLO 这类固定类别检测模型，Grounding DINO 不需要提前限定固定类别，而是可以通过文本 prompt 指定检测目标。

---

## 5. 项目结构

```text
openvision/
├── app.py                    # Gradio 可视化界面
├── demo.py                   # 命令行测试入口
├── requirements.txt          # 项目依赖
├── README.md                 # 项目说明文档
├── data/
│   └── images/               # 测试图片
├── outputs/                  # 输出结果
└── src/
    ├── detector.py           # Grounding DINO 检测模块
    ├── visualizer.py         # 检测框可视化模块
    ├── report.py             # 检测统计报告模块
    ├── postprocess.py        # 标签归一化、置信度过滤和 NMS 去重
    ├── risk_analyzer.py      # 工地安全数量级风险分析
    └── spatial_analyzer.py   # 基于 bbox 空间关系的逐人风险分析
```

---

## 6. 安装依赖

```bash
pip install -r requirements.txt
```

如果在 Google Colab 中运行，建议先开启 GPU：

```text
代码执行程序 → 更改运行时类型 → T4 GPU
```

---

## 7. 命令行运行

```bash
python demo.py
```

运行后会在 `outputs/` 文件夹下生成：

```text
detection_result.jpg
report.json
risk_report.json
spatial_report.json
```

---

## 8. 启动 Gradio 页面

```bash
python app.py
```

如果在 Colab 中运行，需要使用公网链接：

```python
demo.launch(share=True)
```

运行后打开 Gradio 提供的 `gradio.live` 链接即可访问系统界面。

---

## 9. 检测模式说明

### 9.1 通用检测模式

适合检测普通目标，例如：

```text
cat. remote.
car. bus. person.
dog. bicycle.
pig. goat.
```

该模式保留 Grounding DINO 的原始检测结果，不进行工地安全专用过滤。

### 9.2 工地安全检测模式

适合检测工地场景，例如：

```text
person. helmet. hard hat. reflective vest.
```

该模式会进行标签归一化和后处理，例如：

```text
hard hat → helmet
reflective vest → safety vest
```

并进一步生成安全风险提示。

---

## 10. 核心模块说明

### 10.1 detector.py

`detector.py` 负责加载 Grounding DINO 模型，并完成开放词汇目标检测。

主要功能：

- 加载 Hugging Face 上的 Grounding DINO 模型；
- 接收图片路径和文本 prompt；
- 输出检测框、类别标签和置信度。

输出结果格式示例：

```python
{
    "label": "person",
    "score": 0.7497,
    "bbox": [208.64, 75.88, 321.5, 355.62]
}
```

其中：

- `label` 表示检测类别；
- `score` 表示置信度；
- `bbox` 表示检测框坐标，格式为 `[x1, y1, x2, y2]`。

---

### 10.2 visualizer.py

`visualizer.py` 负责将检测结果绘制到原图上。

主要功能：

- 根据 bbox 绘制检测框；
- 在检测框上显示类别和置信度；
- 保存检测结果图片。

输出文件示例：

```text
outputs/detection_result.jpg
```

---

### 10.3 report.py

`report.py` 负责生成结构化检测报告。

主要功能：

- 统计图片尺寸；
- 统计目标总数；
- 统计每个类别的数量；
- 计算检测框面积；
- 计算检测框面积占整张图片的比例；
- 保存 JSON 格式报告。

输出文件示例：

```text
outputs/report.json
```

---

### 10.4 postprocess.py

`postprocess.py` 负责对模型检测结果进行后处理。

主要功能：

- 标签归一化；
- 低置信度过滤；
- NMS 去重。

例如：

```text
hard hat → helmet
reflective vest → safety vest
```

该模块主要用于工地安全检测模式。

---

### 10.5 risk_analyzer.py

`risk_analyzer.py` 负责基于检测数量进行工地安全风险判断。

当前规则：

```text
person 数量 > helmet 数量 → 疑似有人未佩戴安全帽
person 数量 > safety vest 数量 → 疑似有人未穿反光背心
```

该模块属于数量级风险分析，适合作为第一版风险提示。

---

### 10.6 spatial_analyzer.py

`spatial_analyzer.py` 负责基于检测框空间关系进行逐人风险判断。

当前规则：

```text
helmet 的中心点落在 person 头部区域 → 认为该人佩戴安全帽
safety vest 的中心点落在 person 身体区域 → 认为该人穿反光背心
```

该模块可以输出每个人的安全状态，例如：

```json
{
    "person_id": 1,
    "has_helmet": true,
    "has_safety_vest": false,
    "risks": ["未检测到反光背心"]
}
```

---

### 10.7 app.py

`app.py` 使用 Gradio 构建交互式网页界面。

主要功能：

- 上传图片；
- 输入检测 prompt；
- 调节 threshold；
- 调节 text_threshold；
- 选择检测模式；
- 显示检测结果图；
- 显示 JSON 检测报告。

---

## 11. 当前效果与局限

本项目在目标清晰、类别明显的图片中检测效果较好，例如人员、安全帽、车辆、猫、狗等目标。

但在以下场景中仍存在不足：

- 密集小目标检测不稳定；
- 相似类别容易混淆；
- 反光背心等衣物局部属性识别不够稳定；
- 遮挡严重或目标过小时容易漏检；
- prompt 过于宽泛时容易产生误检；
- 当前风险判断主要基于检测框和规则，尚未达到工业级安全检测精度。

---

## 12. 后续优化方向

后续可以从以下方向继续优化：

- 引入 SAM 进行目标分割，获得更精细的 mask；
- 使用更大的 Grounding DINO 模型提升检测能力；
- 针对工地安全场景构建小规模标注数据集；
- 结合颜色特征判断反光背心区域；
- 增加 IoU、中心点距离等更稳健的空间匹配规则；
- 引入 YOLO、RT-DETR 等专用检测模型进行对比实验；
- 增加批量图片检测功能；
- 增加结果下载功能；
- 部署为在线 Web 应用。

---

## 13. 项目亮点

- 使用视觉基础模型完成开放词汇目标检测；
- 支持用户通过文本 prompt 自定义检测目标；
- 将模型检测结果封装成完整工程流程；
- 支持检测结果可视化和 JSON 报告输出；
- 针对工地安全场景设计了风险分析模块；
- 支持通用检测和工地安全检测双模式；
- 使用 Gradio 实现可交互展示界面。

---

## 14. 项目不足说明

本项目主要基于预训练 Grounding DINO 模型进行推理，没有针对特定场景进行微调。因此，系统在开放词汇检测上具有较好的灵活性，但在高精度工业检测场景中仍存在不足。

例如，在工地安全检测中，安全帽相对容易检测，但反光背心属于衣物局部属性，边界不够清晰，容易受到遮挡、角度、颜色和 prompt 表达的影响。因此，当前系统更适合作为开放词汇检测与工程化封装的项目展示，而不是直接用于工业级安全监管。

---

## 15. 项目运行示例

### 通用检测

输入 prompt：

```text
cat. remote.
```

输出：

```text
检测框
类别标签
置信度
JSON 统计报告
```

### 工地安全检测

输入 prompt：

```text
person. helmet. hard hat. reflective vest.
```

输出：

```text
人员检测框
安全帽检测框
反光背心检测框
安全风险提示
逐人风险分析结果
```

---