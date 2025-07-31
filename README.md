# 文献分析应用 (Literature Analyzer)

线上体验→(网址)[https://literature-analyzer-larrtroffen.streamlit.app/]

基于Streamlit的交互式文献分析应用，专门处理从Web of Science (WoS)导出的Full Record Excel文件，旨在为科研人员提供强大的工具来探索大规模文献集合，揭示学科结构、研究热点、以及期刊间的学术内容与风格差异。

## 功能特性

- **批量数据导入与整合**：无缝处理用户上传的多个、零散的WoS Excel文件
- **自动化数据清洗与预处理**：严格按照规约对数据进行清洗、筛选和文本预处理
- **多样化嵌入模型**：支持9种不同的嵌入模型，包括：
  - **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2, text-embedding-ada-002
  - **Gensim模型**: Doc2Vec, Word2Vec, FastText（支持自定义训练）
  - **Hugging Face Transformers**: BERT, RoBERTa, DistilBERT
- **深度文本分析**：对文献的标题和摘要进行语义嵌入、降维、主题建模与聚类
- **全交互式可视化**：使用Plotly构建可交互的图表，全方位展示分析结果

## 技术栈

- **Python环境管理**：`uv`
- **前端/应用框架**：`streamlit`
- **数据处理**：`pandas`, `openpyxl`
- **NLP - 语义嵌入**：`sentence-transformers`, `gensim`, `transformers`
- **NLP - 降维**：`umap-learn`
- **NLP - 主题建模**：`bertopic`
- **NLP - 基础处理**：`nltk`
- **深度学习**：`torch`
- **可视化**：`plotly`, `wordcloud`
- **进度显示**：`tqdm`

## 环境要求

- Python 3.10 或更高版本
- `uv` 包管理器

## 安装与运行

### 1. 安装uv包管理器

```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆项目

```bash
git clone <repository-url>
cd literature-analyzer
```

### 3. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

### 4. 运行应用

```bash
streamlit run app.py
```

## 项目结构

```
.
├── .venv/                  # 由 uv 创建的虚拟环境目录
├── .gitignore              # Git 忽略文件配置
├── app.py                  # Streamlit 主应用文件
├── pyproject.toml          # 项目元数据和依赖项配置文件
├── README.md               # 项目介绍、安装和使用指南
|
├── data/
│   ├── raw/                # 存放原始的、未经处理的数据
│   ├── processed/          # 存放处理后的数据
│   └── sample/             # 存放用于演示或测试的小样本数据
│       └── wos_sample.xlsx
|
├── docs/
│   ├── design_spec.md      # 设计规约文档
│   ├── user_guide.md       # 用户操作手册
│   └── developer_guide.md  # 开发者维护和扩展指南
|
└── src/
    └── literature_analyzer/
        ├── __init__.py
        ├── data_processing.py  # 数据导入、合并、清洗和预处理
        ├── nlp_analysis.py     # 语义嵌入、降维和主题建模
        └── visualization.py    # 生成所有Plotly图表和词云
```

## 使用说明

1. **数据上传**：在左侧控制面板上传一个或多个WoS Excel文件
2. **数据处理**：点击"1. 加载并处理数据"按钮进行数据清洗和预处理
3. **嵌入与降维**：选择嵌入模型和UMAP参数，点击"2. 执行嵌入与降维"
   - **快速分析**：选择all-MiniLM-L6-v2（轻量级，速度快）
   - **平衡分析**：选择all-mpnet-base-v2（中等性能，质量好）
   - **深度分析**：选择BERT、RoBERTa或text-embedding-ada-002（高质量，速度慢）
   - **自定义训练**：选择Doc2Vec、Word2Vec或FastText（基于您的数据训练）
4. **主题分析**：配置BERTopic参数，点击"3. 执行主题分析"
5. **结果探索**：在不同标签页中查看分析结果：
   - 数据概览：查看处理后的数据和统计信息
   - 语义空间探索：3D散点图展示文献语义分布
   - 主题分析：主题分布和词云
   - 期刊对比：期刊-主题分布对比
   - 时间趋势：主题随时间演变趋势

## 文档

详细的使用说明和开发文档请参考 `docs/` 目录：

- `docs/design_spec.md` - 详细的设计规约
- `docs/user_guide.md` - 用户操作手册
- `docs/developer_guide.md` - 开发者指南

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
