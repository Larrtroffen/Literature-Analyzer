# 开发者维护和扩展指南

## 概述

本文档面向文献分析应用的开发者和维护者，提供代码架构说明、开发环境配置、功能扩展指南以及维护最佳实践。遵循本文档的指导可以确保代码质量和项目的可持续发展。

## 开发环境配置

### 1. 环境要求

- Python 3.10+
- uv 包管理器
- Git
- VS Code（推荐）或其他IDE

### 2. 项目设置

```bash
# 克隆项目
git clone <repository-url>
cd literature-analyzer

# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖（包含开发依赖）
uv pip install -e ".[dev]"

# 下载spaCy英语模型
python -m spacy download en_core_web_sm
```

### 3. 开发工具配置

#### VS Code推荐扩展
- Python
- Pylance
- Black Formatter
- GitLens

#### 代码格式化
项目使用Black进行代码格式化：

```bash
# 格式化代码
black src/ app.py

# 检查格式化
black --check src/ app.py
```

#### 类型检查
使用MyPy进行静态类型检查：

```bash
mypy src/literature_analyzer/
```

#### 代码质量检查
使用Flake8进行代码质量检查：

```bash
flake8 src/ app.py
```

## 项目架构详解

### 目录结构

```
src/literature_analyzer/
├── __init__.py           # 包初始化文件
├── data_processing.py    # 数据处理模块
├── nlp_analysis.py       # NLP分析模块
└── visualization.py      # 可视化模块
```

### 模块职责

#### 1. data_processing.py
**职责**：数据导入、清洗、预处理
**核心类/函数**：
- `load_and_process_data()`：主要数据处理函数
- `preprocess_text()`：文本预处理函数
- `clean_dataframe()`：数据清洗函数

**设计原则**：
- 纯函数设计，避免副作用
- 详细的错误处理和日志记录
- 支持多种数据格式

#### 2. nlp_analysis.py
**职责**：语义嵌入、降维、主题建模
**核心类/函数**：
- `generate_embeddings()`：生成文本嵌入
- `perform_umap()`：UMAP降维
- `perform_topic_modeling()`：BERTopic主题建模

**设计原则**：
- 使用Streamlit缓存机制
- 模型加载的资源管理
- 参数验证和默认值处理

#### 3. visualization.py
**职责**：生成各种可视化图表
**核心类/函数**：
- `create_3d_scatter()`：3D散点图
- `create_topic_distribution()`：主题分布图
- `create_wordcloud()`：词云生成
- `create_journal_comparison()`：期刊对比图
- `create_temporal_trends()`：时间趋势图

**设计原则**：
- 图表配置的灵活性
- 交互功能的实现
- 性能优化（大数据集处理）

### app.py 架构

**主要组件**：
1. **会话状态管理**：使用`st.session_state`管理应用状态
2. **UI布局**：侧边栏控制面板 + 主内容区域
3. **流程控制**：三步分析流程的状态管理
4. **缓存管理**：智能缓存策略

**关键设计模式**：
- 状态机模式：管理分析流程的不同阶段
- 观察者模式：UI控件与数据的同步更新
- 策略模式：不同分析算法的切换

## 开发规范

### 1. 代码风格

- 遵循PEP 8规范
- 使用Black自动格式化
- 最大行长度88字符
- 使用类型注解

### 2. 命名约定

```python
# 函数名：小写下划线
def process_data():
    pass

# 类名：驼峰命名
class DataProcessor:
    pass

# 变量名：小写下划线
processed_data = []

# 常量：大写下划线
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
```

### 3. 文档字符串

使用Google风格的docstring：

```python
def process_text(text: str) -> str:
    """
    对输入文本进行预处理，包括清洗、标准化和分词。
    
    Args:
        text (str): 需要处理的原始文本
        
    Returns:
        str: 处理后的文本
        
    Raises:
        ValueError: 当输入文本为空或无效时
        
    Example:
        >>> process_text("Hello World!")
        'hello world'
    """
    if not text or not isinstance(text, str):
        raise ValueError("输入文本必须是非空字符串")
    
    # 处理逻辑
    return text.lower().strip()
```

### 4. 错误处理

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_operation(data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    安全地执行可能失败的操作。
    
    Args:
        data: 输入数据，可能为None
        
    Returns:
        处理后的数据
        
    Raises:
        ProcessingError: 当处理失败时
    """
    try:
        if data is None:
            raise ValueError("输入数据不能为None")
            
        # 执行操作
        result = data.copy()
        
        return result
        
    except Exception as e:
        logger.error(f"操作失败: {str(e)}")
        raise ProcessingError(f"数据处理失败: {str(e)}") from e
```

## 功能扩展指南

### 1. 添加新的数据处理功能

#### 步骤1：在data_processing.py中添加函数

```python
def custom_data_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    自定义数据转换函数。
    
    Args:
        df: 输入DataFrame
        
    Returns:
        转换后的DataFrame
    """
    # 实现自定义逻辑
    transformed_df = df.copy()
    
    # 添加处理逻辑
    transformed_df['new_column'] = transformed_df['existing_column'].apply(
        lambda x: custom_function(x)
    )
    
    return transformed_df
```

#### 步骤2：更新主处理流程

```python
def load_and_process_data(uploaded_files):
    # ... 现有代码 ...
    
    # 添加自定义处理步骤
    df = custom_data_transform(df)
    
    return df
```

#### 步骤3：添加UI控件（如需要）

```python
# 在app.py中添加
enable_custom_transform = st.checkbox("启用自定义转换")
```

### 2. 添加新的NLP模型

#### 步骤1：在nlp_analysis.py中添加模型支持

```python
# 添加到模型列表
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "custom-model": "your-custom-model-name",
}

def generate_custom_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """
    使用自定义模型生成嵌入向量。
    
    Args:
        texts: 文本列表
        model_name: 模型名称
        
    Returns:
        嵌入向量矩阵
    """
    model = load_embedding_model(model_name)
    embeddings = model.encode(texts)
    return embeddings
```

#### 步骤2：更新UI模型选择

```python
# 在app.py中更新模型选择选项
model_options = list(AVAILABLE_MODELS.keys())
selected_model = st.selectbox("选择嵌入模型", model_options)
```

### 3. 添加新的可视化类型

#### 步骤1：在visualization.py中添加图表函数

```python
def create_custom_chart(data: pd.DataFrame, **kwargs) -> go.Figure:
    """
    创建自定义图表。
    
    Args:
        data: 输入数据
        **kwargs: 图表配置参数
        
    Returns:
        Plotly图表对象
    """
    fig = px.custom_plot(
        data,
        x=kwargs.get('x_column'),
        y=kwargs.get('y_column'),
        color=kwargs.get('color_column'),
        title=kwargs.get('title', '自定义图表')
    )
    
    # 自定义图表配置
    fig.update_layout(
        showlegend=True,
        hovermode='closest'
    )
    
    return fig
```

#### 步骤2：在主应用中集成

```python
# 在app.py中添加新的标签页
with st.tabs(["...", "自定义视图"]):
    if st.session_state.get('analysis_complete'):
        custom_fig = create_custom_chart(
            processed_data,
            x_column='x',
            y_column='y',
            color_column='topic_name'
        )
        st.plotly_chart(custom_fig, use_container_width=True)
```

### 4. 添加新的数据源支持

#### 步骤1：创建数据源适配器

```python
class DataSourceAdapter:
    """数据源适配器基类"""
    
    def load_data(self, source) -> pd.DataFrame:
        """加载数据的抽象方法"""
        raise NotImplementedError
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据格式"""
        raise NotImplementedError

class WoSExcelAdapter(DataSourceAdapter):
    """WoS Excel数据源适配器"""
    
    def load_data(self, file) -> pd.DataFrame:
        """加载WoS Excel文件"""
        return pd.read_excel(file)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证WoS数据格式"""
        required_columns = ['Article Title', 'Source Title', 'Publication Year', 'Abstract']
        return all(col in df.columns for col in required_columns)

class CustomCSVAdapter(DataSourceAdapter):
    """自定义CSV数据源适配器"""
    
    def load_data(self, file) -> pd.DataFrame:
        """加载CSV文件"""
        return pd.read_csv(file)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证CSV数据格式"""
        # 自定义验证逻辑
        return True
```

#### 步骤2：更新数据加载逻辑

```python
def load_data_from_source(uploaded_files, source_type: str):
    """
    根据数据源类型加载数据。
    
    Args:
        uploaded_files: 上传的文件列表
        source_type: 数据源类型 ('wos_excel', 'custom_csv')
        
    Returns:
        合并后的DataFrame
    """
    if source_type == 'wos_excel':
        adapter = WoSExcelAdapter()
    elif source_type == 'custom_csv':
        adapter = CustomCSVAdapter()
    else:
        raise ValueError(f"不支持的数据源类型: {source_type}")
    
    dataframes = []
    for file in uploaded_files:
        df = adapter.load_data(file)
        if adapter.validate_data(df):
            dataframes.append(df)
        else:
            logger.warning(f"文件 {file.name} 格式验证失败")
    
    if not dataframes:
        raise ValueError("没有有效的数据文件")
    
    return pd.concat(dataframes, ignore_index=True)
```

## 性能优化指南

### 1. 缓存策略

#### 数据缓存
```python
@st.cache_data(ttl=3600, max_entries=10)
def cached_data_processing(files_hash, processing_params):
    """
    缓存数据处理结果。
    
    Args:
        files_hash: 文件哈希值
        processing_params: 处理参数
        
    Returns:
        处理后的数据
    """
    return load_and_process_data_internal(files_hash, processing_params)
```

#### 模型缓存
```python
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    """
    缓存嵌入模型加载。
    
    Args:
        model_name: 模型名称
        
    Returns:
        加载的模型
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
```

### 2. 内存管理

#### 大数据集处理
```python
def process_large_dataset(df: pd.DataFrame, chunk_size: int = 1000):
    """
    分块处理大数据集。
    
    Args:
        df: 输入数据
        chunk_size: 块大小
        
    Yields:
        处理后的数据块
    """
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        yield process_chunk(chunk)
        
        # 手动触发垃圾回收
        import gc
        gc.collect()
```

#### 内存监控
```python
import psutil

def monitor_memory_usage():
    """监控内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    st.sidebar.metric(
        "内存使用",
        f"{memory_info.rss / 1024 / 1024:.1f} MB"
    )
    
    if memory_info.rss > 2 * 1024 * 1024 * 1024:  # 2GB
        st.warning("内存使用过高，建议清除缓存或减少数据量")
```

### 3. 异步处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_process_data(data):
    """异步处理数据"""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            process_data_sync,
            data
        )
    
    return result

def process_data_sync(data):
    """同步数据处理函数"""
    # 耗时的处理逻辑
    return processed_data
```

## 测试指南

### 1. 单元测试

创建`tests/`目录并添加测试文件：

```python
# tests/test_data_processing.py
import pytest
import pandas as pd
from src.literature_analyzer.data_processing import preprocess_text

def test_preprocess_text():
    """测试文本预处理功能"""
    input_text = "Hello World! This is a TEST."
    expected = "hello world test"
    
    result = preprocess_text(input_text)
    assert result == expected

def test_preprocess_empty_text():
    """测试空文本处理"""
    with pytest.raises(ValueError):
        preprocess_text("")
```

### 2. 集成测试

```python
# tests/test_integration.py
def test_full_analysis_workflow():
    """测试完整的分析工作流"""
    # 准备测试数据
    test_data = create_test_data()
    
    # 执行完整流程
    processed_data = load_and_process_data(test_data)
    embeddings = generate_embeddings(processed_data)
    topics = perform_topic_modeling(processed_data, embeddings)
    
    # 验证结果
    assert len(processed_data) > 0
    assert embeddings.shape[0] == len(processed_data)
    assert len(topics) > 0
```

### 3. 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_data_processing.py

# 生成覆盖率报告
pytest --cov=src/literature_analyzer tests/
```

## 部署指南

### 1. Docker部署

创建`Dockerfile`：

```dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN pip install uv

# 复制项目文件
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY app.py ./

# 安装依赖
RUN uv pip install -e .

# 暴露端口
EXPOSE 8501

# 运行应用
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. 云服务部署

#### Streamlit Community Cloud
1. 将代码推送到GitHub仓库
2. 在Streamlit Cloud中连接仓库
3. 配置Python版本和依赖
4. 部署应用

#### AWS/Azure/GCP
```bash
# 构建Docker镜像
docker build -t literature-analyzer .

# 运行容器
docker run -p 8501:8501 literature-analyzer
```

## 维护指南

### 1. 依赖管理

#### 更新依赖
```bash
# 检查过时的依赖
uv pip list --outdated

# 更新特定依赖
uv pip install package_name --upgrade

# 更新所有依赖
uv pip sync -U
```

#### 依赖安全检查
```bash
# 安装安全检查工具
uv pip install safety

# 检查安全漏洞
safety check
```

### 2. 版本管理

#### 语义化版本控制
- 主版本号：不兼容的API更改
- 次版本号：向下兼容的功能新增
- 修订号：向下兼容的问题修复

#### 发布流程
```bash
# 更新版本号
# 在pyproject.toml中更新version字段

# 创建标签
git tag -a v2.1.0 -m "Version 2.1.0"

# 推送标签
git push origin v2.1.0

# 构建发布包
uv build
```

### 3. 监控和日志

#### 应用监控
```python
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_usage_stats():
    """记录使用统计"""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.get('session_id'),
        'data_size': st.session_state.get('data_size', 0),
        'analysis_steps': st.session_state.get('analysis_steps', [])
    }
    logger.info(f"Usage stats: {stats}")
```

### 4. 文档维护

#### 自动生成API文档
```bash
# 安装Sphinx
uv pip install sphinx sphinx-rtd-theme

# 生成文档
cd docs/
sphinx-apidoc -o . ../src/literature_analyzer/
make html
```

#### 保持文档同步
- 代码变更时同步更新相关文档
- 使用自动化工具检查文档与代码的一致性
- 定期审查和更新用户指南

## 故障排除

### 1. 常见问题

#### 内存不足
```python
# 在app.py中添加内存检查
def check_memory_usage():
    """检查内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_percent = process.memory_percent()
    
    if memory_percent > 80:
        st.error("内存使用过高，请清除缓存或减少数据量")
        return False
    return True
```

#### 模型加载失败
```python
# 添加模型加载重试机制
def load_model_with_retry(model_name, max_retries=3):
    """带重试机制的模型加载"""
    from sentence_transformers import SentenceTransformer
    import time
    
    for attempt in range(max_retries):
        try:
            model = SentenceTransformer(model_name)
            return model
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 指数退避
```

### 2. 调试技巧

#### Streamlit调试
```python
# 添加调试信息
if st.secrets.get("DEBUG", False):
    st.write("调试信息:")
    st.write(f"Session state: {st.session_state}")
    st.write(f"Data shape: {data.shape if 'data' in locals() else 'No data'}")
```

#### 性能分析
```python
import cProfile
import pstats

def profile_function(func):
    """性能分析装饰器"""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    return wrapper
```

## 贡献指南

### 1. 代码贡献流程

1. Fork项目仓库
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 编写代码和测试
4. 确保代码通过所有检查：`black . && flake8 . && mypy . && pytest`
5. 提交变更：`git commit -m "feat: add new feature"`
6. 推送分支：`git push origin feature/new-feature`
7. 创建Pull Request

### 2. 提交信息规范

使用Conventional Commits格式：
```
feat: 添加新功能
fix: 修复bug
docs: 文档更新
style: 代码格式化
refactor: 代码重构
test: 测试相关
chore: 构建或辅助工具变动
```

### 3. 代码审查清单

- [ ] 代码符合项目风格指南
- [ ] 所有测试通过
- [ ] 添加了必要的文档
- [ ] 性能影响已评估
- [ ] 安全影响已考虑
- [ ] 向后兼容性已考虑

---

*本文档将随着项目的发展持续更新，建议定期查看最新版本。*
