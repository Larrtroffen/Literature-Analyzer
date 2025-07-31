# 文献分析应用：工程设计与实现规约 (v2.0)

## 1. 应用概述

**目标：** 开发一个基于Streamlit的交互式文献分析应用。该应用专门处理从Web of Science (WoS)导出的Full Record Excel文件，旨在为科研人员提供一个强大的工具，用于探索大规模文献集合，揭示学科结构、研究热点、以及期刊间的学术内容与风格差异。

**核心功能：**
1.  **批量数据导入与整合：** 无缝处理用户上传的多个、零散的WoS Excel文件。
2.  **自动化数据清洗与预处理：** 严格按照规约对数据进行清洗、筛选和文本预处理。
3.  **深度文本分析：** 对文献的标题和摘要进行语义嵌入、降维、主题建模与聚类。
4.  **全交互式可视化：** 使用Plotly构建可交互的图表，全方位展示分析结果。

**目标用户：** 需要进行文献综述、学科趋势分析、期刊评估的科研人员、研究生及图书情报专业人员。

## 2. 系统架构与工程规约

### 2.1. 技术栈

*   **Python环境管理：** `uv`
*   **前端/应用框架：** `streamlit`
*   **数据处理：** `pandas`, `openpyxl`
*   **NLP - 语义嵌入：** `sentence-transformers`
*   **NLP - 降维：** `umap-learn`
*   **NLP - 主题建模：** `bertopic`
*   **NLP - 基础处理：** `spacy`
*   **可视化：** `plotly`, `wordcloud`
*   **进度显示：** `tqdm`

### 2.2. 项目目录结构

项目目录结构经过精心设计，以分离代码、文档、数据和配置，确保项目的清晰度和可维护性。

```
.
├── .venv/                  # 由 uv 创建的虚拟环境目录
├── .gitignore              # Git 忽略文件配置
├── app.py                  # Streamlit 主应用文件，负责UI布局和流程控制
├── pyproject.toml          # 项目元数据和依赖项配置文件 (uv 使用)
├── README.md               # 项目介绍、安装和使用指南
|
├── data/
│   ├── raw/                # 存放原始的、未经处理的数据 (Git 忽略)
│   │   └── .gitkeep
│   ├── processed/          # 存放处理后的数据 (Git 忽略)
│   │   └── .gitkeep
│   └── sample/             # 存放用于演示或测试的小样本数据
│       └── wos_sample.xlsx
|
├── docs/
│   ├── design_spec.md      # 本设计规约文档
│   ├── user_guide.md       # 用户操作手册
│   └── developer_guide.md  # 开发者维护和扩展指南
|
└── src/
    └── literature_analyzer/
        ├── __init__.py
        ├── data_processing.py  # 模块1：负责数据导入、合并、清洗和预处理
        ├── nlp_analysis.py     # 模块2：负责语义嵌入、降维和主题建模
        └── visualization.py    # 模块3：负责生成所有Plotly图表和词云
```

**目录说明：**
*   **`uv`管理：** `uv`将使用`pyproject.toml`来管理依赖，并在`.venv`目录中创建隔离的虚拟环境。
*   **`data/`：**
    *   `raw/`和`processed/`目录用于存放用户上传和应用处理的数据，这些目录应被添加到`.gitignore`中，以避免将大数据文件提交到版本库。
    *   `sample/`目录包含一个小的示例文件，方便新用户快速上手和开发者进行测试。
*   **`docs/`：**
    *   `design_spec.md`：即本文档，作为项目开发的蓝图。
    *   `user_guide.md`：面向最终用户的操作说明，解释如何上传数据、如何与图表交互等。
    *   `developer_guide.md`：面向未来维护者的技术文档，解释代码架构、如何添加新功能、如何更新依赖等。
*   **`src/literature_analyzer/`：**
    *   所有核心业务逻辑代码被封装在一个名为`literature_analyzer`的Python包中，位于`src`目录下。这是一种现代Python项目的标准实践，有助于命名空间管理和打包分发。`app.py`将从这个包中导入功能模块。

### 2.3. 工程核心原则

1.  **环境管理 (`uv`)：**
    *   **初始化：** `uv venv` 创建虚拟环境。
    *   **激活：** `source .venv/bin/activate` (Linux/macOS) 或 `.venv\Scripts\activate` (Windows)。
    *   **安装依赖：** `uv pip install -r requirements.txt` 或直接 `uv pip install streamlit pandas ...`。依赖项将记录在`pyproject.toml`中。
    *   **`requirements.txt`生成：** `uv pip freeze > requirements.txt` 用于生成明确的版本锁定文件。
2.  **模块化与单一职责：** 每个`.py`文件和函数都应有明确的单一职责。
3.  **缓存策略：**
    *   **`@st.cache_resource`：** 必须用于加载计算资源密集型对象，如`sentence-transformers`模型。
    *   **`@st.cache_data`：** 必须用于缓存计算结果，如清洗后的DataFrame、嵌入向量、降维坐标和训练好的主题模型。
4.  **状态管理：** 必须使用`st.session_state`来存储贯穿用户会话的变量。
5.  **错误处理与用户反馈：** 所有可能失败的操作都必须包含在`try-except`块中，并向用户显示明确的错误信息。所有耗时操作都必须使用`st.spinner()`包裹。
6.  **文档与注释 (Docstrings)：** 项目中的每一个函数都必须包含符合Google Python Style Guide或NumPy/SciPy Docstring Standard的文档字符串。

## 3. 核心功能模块详细规约

### 3.1. 模块一：数据导入与预处理 (`src/literature_analyzer/data_processing.py`)

**输入：** 用户通过`st.file_uploader`上传的一个或多个WoS Excel文件。
**输出：** 一个经过完全清洗和预处理的`pandas.DataFrame`。

**处理流程（严格按序执行）：**

1.  **文件合并：**
    *   遍历用户上传的所有文件。
    *   使用`pd.read_excel()`读取每个文件。
    *   将所有DataFrame使用`pd.concat(ignore_index=True)`合并为一个。
2.  **列筛选与重命名：**
    *   仅保留以下列，并重命名为内部标准名称：
        *   `'Article Title'` -> `article_title`
        *   `'Source Title'` -> `journal_title`
        *   `'Publication Year'` -> `publication_year`
        *   `'Abstract'` -> `abstract_text`
3.  **数据清洗：**
    *   **去重：** 基于`article_title`和`abstract_text`两列的组合，去除完全重复的记录。
    *   **缺失值处理：** 严格删除任何在`article_title`, `journal_title`, `publication_year`, `abstract_text`中包含缺失值的行。
    *   **类型转换：** 将`publication_year`列转换为整数类型。
    *   **内容过滤：** 删除`abstract_text`列中字符数少于100的记录。
4.  **文本预处理：**
    *   **合并文本：** 创建一个新列`full_text`，其内容为`article_title`和`abstract_text`的拼接。
    *   **创建`processed_text`列：** 对`full_text`列应用以下连续处理步骤：
        1.  **规范化：** 转换为小写，移除所有URL和HTML标签。
        2.  **标点与数字移除：** 移除所有标点符号和数字。
        3.  **停用词移除与词形还原：** 使用`spacy`的`en_core_web_sm`模型进行分词、停用词移除、词性标注和词形还原。
        5.  **空格处理：** 移除处理后文本中多余的空格。

### 3.2. 模块二：NLP分析 (`src/literature_analyzer/nlp_analysis.py`)

**输入：** 预处理后的DataFrame，特别是`processed_text`列。
**输出：** 包含嵌入坐标、主题信息等新列的DataFrame，以及训练好的主题模型。

**处理流程：**

1.  **语义嵌入：**
    *   **模型：** 用户从UI选择一个预定义的`sentence-transformers`模型。
    *   **执行：** 将`processed_text`列的所有文本编码为高维向量。
2.  **降维：**
    *   **算法：** 使用UMAP算法。
    *   **配置：** 用户通过UI滑块设置`n_neighbors`和`min_dist`参数。目标维度由用户选择（2D或3D）。
    *   **执行：** 将高维嵌入向量降维至2D或3D坐标。
3.  **主题建模：**
    *   **算法：** 使用BERTopic算法。
    *   **输入：** `processed_text`列表和预先计算的嵌入向量。
    *   **配置：** 用户通过UI设置`min_topic_size`和`nr_topics`（或选择`auto`）。
    *   **执行：** 训练BERTopic模型。
    *   **结果处理：**
        *   将主题ID为-1的 outlier 主题重命名为"Unclassified"。
        *   将主题ID、主题名称、主题概率添加回DataFrame。

### 3.3. 模块三：交互式可视化 (`src/literature_analyzer/visualization.py`)

**输入：** 包含所有分析结果的最终DataFrame。
**输出：** Plotly图表对象和WordCloud图像。

**可视化组件规约：**

1.  **语义空间散点图 (3D)：**
    *   **图表类型：** `plotly.express.scatter_3d`。
    *   **坐标轴：** `x`, `y`, `z`。
    *   **颜色映射：** 用户可选择按`topic_name`, `journal_title`, 或 `publication_year`着色。
    *   **悬停信息：** 鼠标悬停时必须显示`article_title`, `journal_title`, `publication_year`, `topic_name`。
    *   **期刊覆盖范围：** 对于用户在UI中选择的期刊，计算其所有文章在3D空间中的凸包（使用`scipy.spatial.ConvexHull`），并在图上用半透明的网格或线条绘制出来。
    *   **期刊中心点：** 计算每个选定期刊文章坐标的平均值，并在图上用一个独特的、较大的标记（如星形）标出。
2.  **主题分析视图：**
    *   **主题分布图：** 使用`plotly.express.bar`绘制条形图，显示每个主题包含的文章数量。
    *   **主题词云：** 为每个主题（不包括"Unclassified"）生成一个词云。
3.  **期刊对比视图：**
    *   **期刊-主题分布图：** 使用`plotly.express.bar`绘制堆叠归一化条形图。
4.  **时间趋势视图：**
    *   **主题演变图：** 使用`plotly.express.line`绘制折线图。

## 4. 用户界面(UI)布局

UI将采用双栏布局，左侧为控制面板，右侧为主内容展示区。

### 4.1. 左侧边栏 (控制面板)

*   **第一部分：数据与分析流程**
    *   **标题：** `数据加载与分析`
    *   **控件1：** `st.file_uploader` 用于上传WoS文件。
    *   **控件2：** `st.button("1. 加载并处理数据")`
    *   **控件3：** `st.selectbox` 用于选择嵌入模型。
    *   **控件4 & 5：** `st.slider` 用于配置UMAP参数。
    *   **控件6：** `st.button("2. 执行嵌入与降维")`
    *   **控件7 & 8：** `st.slider` 用于配置BERTopic参数。
    *   **控件9：** `st.button("3. 执行主题分析")`
*   **第二部分：图表筛选与交互**
    *   **标题：** `图表筛选`
    *   **控件1：** `st.slider` 用于筛选`publication_year`范围。
    *   **控件2：** `st.multiselect` 用于筛选`journal_title`。
    *   **控件3：** `st.multiselect` 用于筛选`topic_name`。
    *   **控件4：** `st.text_input` 用于关键词搜索。
*   **第三部分：工具**
    *   **标题：** `工具`
    *   **控件1：** `st.button("清除所有缓存并重置")`

### 4.2. 右侧主区域 (内容展示)

使用`st.tabs`来组织不同的分析视图。

*   **Tab 1: "数据概览"**
    *   显示处理前后的记录数统计。
    *   显示处理后DataFrame的前10行 (`st.dataframe`)。
    *   提供`st.download_button`以下载完整的、包含所有分析结果的CSV文件。
*   **Tab 2: "语义空间探索"**
    *   显示3D语义空间散点图。
    *   图表上方提供`st.selectbox`让用户选择颜色映射的依据。
*   **Tab 3: "主题分析"**
    *   并排或上下布局显示主题分布条形图和主题词云。
    *   提供一个`st.selectbox`让用户选择要查看词云的主题。
*   **Tab 4: "期刊对比"**
    *   显示期刊-主题分布的堆叠条形图。
*   **Tab 5: "时间趋势"**
    *   显示主题演变折线图。

## 5. 部署与维护

*   **部署：** 推荐使用Docker将应用容器化。`Dockerfile`将基于Python官方镜像，使用`uv`安装`pyproject.toml`中定义的依赖。
*   **维护：**
    *   **依赖管理：** `pyproject.toml`是唯一的依赖来源。定期运行`uv pip sync`来确保环境与配置文件一致。
    *   **版本控制：** 使用Git进行版本控制，所有代码变更都应有清晰的提交信息。
    *   **文档同步：** 当代码逻辑发生变化时，必须同步更新`docs/`目录下的相关文档。
