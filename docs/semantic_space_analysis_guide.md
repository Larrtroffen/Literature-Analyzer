# 高级语义空间分析功能指南

本指南详细介绍了文献分析器中新添加的高级语义空间分析功能，包括多维度降维对比、语义相似性网络、语义演化分析、语义聚类和漂移检测等。

## 目录

1. [功能概述](#功能概述)
2. [核心类介绍](#核心类介绍)
3. [安装和依赖](#安装和依赖)
4. [快速开始](#快速开始)
5. [详细使用说明](#详细使用说明)
   - [SemanticSpaceAnalyzer](#semanticspaceanalyzer)
   - [TemporalSemanticAnalyzer](#temporalsemanticanalyzer)
   - [InteractiveSemanticExplorer](#interactivesemanticexplorer)
   - [可视化函数](#可视化函数)
6. [示例代码](#示例代码)
7. [最佳实践](#最佳实践)
8. [常见问题](#常见问题)

## 功能概述

高级语义空间分析功能提供了以下核心能力：

### 1. 多维度降维对比
- 支持UMAP、t-SNE、PCA等多种降维方法
- 可视化对比不同降维方法的效果
- 帮助选择最适合数据的降维方法

### 2. 语义相似性网络
- 基于余弦相似性构建文档关系网络
- 交互式网络可视化（使用pyvis）
- 可调节相似性阈值控制网络密度

### 3. 语义演化分析
- 跟踪语义空间随时间的变化
- 计算语义漂移指标
- 创建演化动画展示变化过程

### 4. 语义聚类和漂移检测
- 支持多种聚类算法（KMeans、DBSCAN、Agglomerative）
- 自动确定最优聚类数量
- 计算聚类评估指标（轮廓系数、Calinski-Harabasz指数等）

### 5. 交互式3D可视化
- 创建交互式3D语义空间投影
- 支持多种视角切换
- 可按类别或元数据着色

### 6. 语义密度分析
- 生成语义密度热力图
- 识别语义空间中的密集区域
- 可视化数据分布模式

## 核心类介绍

### SemanticSpaceAnalyzer
语义空间分析的核心类，提供：
- 相似性矩阵计算
- 多种降维方法
- 聚类分析
- 语义漂移分析
- 统计信息获取

### TemporalSemanticAnalyzer
时间序列语义分析类，专门用于：
- 管理多个时间步的语义空间
- 分析语义演化趋势
- 计算稳定性指标
- 创建演化动画

### InteractiveSemanticExplorer
交互式语义空间探索器，支持：
- 3D语义空间投影
- 语义搜索功能
- 相似文档查找
- 文档相似性网络构建

## 安装和依赖

### 系统要求
- Python 3.10
- 至少8GB RAM（推荐16GB）
- 支持GPU加速（可选）

### 安装依赖
```bash
pip install -e .
```

### 新增依赖包
- `dash>=2.14.0` - 交互式Web应用框架
- `igraph>=0.11.0` - 高性能图分析库
- `leidenalg>=0.10.0` - 社区检测算法
- `statsmodels>=0.14.0` - 统计模型
- `prophet>=1.1.0` - 时间序列预测
- `optuna>=3.3.0` - 超参数优化
- `shap>=0.42.0` - 模型解释性
- `dask>=2023.10.0` - 并行计算
- `numba>=0.58.0` - JIT编译加速

## 快速开始

### 基础使用
```python
from literature_analyzer import SemanticSpaceAnalyzer
import numpy as np

# 假设已有嵌入向量
embeddings = np.random.rand(100, 384)  # 100个文档，384维

# 创建分析器
analyzer = SemanticSpaceAnalyzer(embeddings)

# 计算相似性矩阵
similarity_matrix = analyzer.compute_similarity_matrix()

# 降维
reduced_embeddings = analyzer.reduce_dimensions(method='UMAP')

# 聚类分析
cluster_labels = analyzer.perform_clustering(method='KMeans')

# 获取统计信息
stats = analyzer.get_semantic_statistics()
print(stats)
```

### 时间序列分析
```python
from literature_analyzer import TemporalSemanticAnalyzer

# 创建时间序列分析器
temporal_analyzer = TemporalSemanticAnalyzer()

# 添加多个时间步的数据
for year, embeddings in time_series_data.items():
    temporal_analyzer.add_time_step(year, embeddings)

# 分析演化
evolution_results = temporal_analyzer.analyze_temporal_evolution()

# 创建动画
animation_fig = temporal_analyzer.create_evolution_animation()
animation_fig.show()
```

## 详细使用说明

### SemanticSpaceAnalyzer

#### 初始化
```python
analyzer = SemanticSpaceAnalyzer(
    embeddings=embeddings,           # 嵌入向量矩阵 (n_samples, n_features)
    labels=document_titles          # 文档标题列表 (可选)
)
```

#### 主要方法

##### 1. compute_similarity_matrix()
计算文档间的余弦相似性矩阵。

```python
similarity_matrix = analyzer.compute_similarity_matrix()
# 返回: (n_samples, n_samples) 的相似性矩阵
```

##### 2. reduce_dimensions()
对嵌入向量进行降维。

```python
# 使用UMAP降维到2D
reduced_embeddings = analyzer.reduce_dimensions(
    method='UMAP',
    n_components=2,
    min_dist=0.1,
    n_neighbors=15
)

# 使用t-SNE降维
reduced_embeddings = analyzer.reduce_dimensions(
    method='t-SNE',
    n_components=2,
    perplexity=30
)

# 使用PCA降维
reduced_embeddings = analyzer.reduce_dimensions(
    method='PCA',
    n_components=2
)
```

##### 3. perform_clustering()
对嵌入向量进行聚类分析。

```python
# KMeans聚类（自动确定聚类数）
cluster_labels = analyzer.perform_clustering(method='KMeans')

# 指定聚类数
cluster_labels = analyzer.perform_clustering(
    method='KMeans',
    n_clusters=5
)

# DBSCAN聚类
cluster_labels = analyzer.perform_clustering(
    method='DBSCAN',
    eps=0.5,
    min_samples=5
)

# 层次聚类
cluster_labels = analyzer.perform_clustering(
    method='Agglomerative',
    n_clusters=5
)
```

##### 4. analyze_semantic_drift()
分析与另一个语义空间的漂移。

```python
# 质心漂移分析
drift_metrics = analyzer.analyze_semantic_drift(
    other_embeddings=other_embeddings,
    method='centroid'
)

# 分布漂移分析
drift_metrics = analyzer.analyze_semantic_drift(
    other_embeddings=other_embeddings,
    method='distribution'
)

# 成对距离分析
drift_metrics = analyzer.analyze_semantic_drift(
    other_embeddings=other_embeddings,
    method='pairwise'
)
```

##### 5. create_semantic_network()
创建语义相似性网络图。

```python
# 创建网络图
network_html = analyzer.create_semantic_network(
    threshold=0.7,    # 相似性阈值
    max_nodes=100     # 最大节点数
)

# 保存到文件
with open('semantic_network.html', 'w') as f:
    f.write(network_html)
```

##### 6. create_density_heatmap()
创建语义密度热力图。

```python
# 创建密度热力图
density_fig = analyzer.create_density_heatmap(
    grid_size=50    # 网格大小
)
density_fig.show()
```

##### 7. get_semantic_statistics()
获取语义空间统计信息。

```python
stats = analyzer.get_semantic_statistics()
# 返回包含以下信息的字典：
# - embedding_shape: 嵌入向量形状
# - embedding_dimension: 嵌入维度
# - num_documents: 文档数量
# - mean_similarity: 平均相似度
# - std_similarity: 相似度标准差
# - 聚类相关指标（如果已执行聚类）
```

### TemporalSemanticAnalyzer

#### 初始化
```python
temporal_analyzer = TemporalSemanticAnalyzer()
```

#### 主要方法

##### 1. add_time_step()
添加时间步数据。

```python
temporal_analyzer.add_time_step(
    time_step=2020,           # 时间步标识
    embeddings=embeddings,     # 嵌入向量
    labels=document_titles    # 文档标题（可选）
)
```

##### 2. analyze_temporal_evolution()
分析时间演化。

```python
evolution_results = temporal_analyzer.analyze_temporal_evolution()
# 返回包含以下信息的字典：
# - time_steps: 时间步列表
# - drift_metrics: 漂移指标
# - stability_metrics: 稳定性指标
```

##### 3. create_evolution_animation()
创建语义空间演化动画。

```python
animation_fig = temporal_analyzer.create_evolution_animation()
animation_fig.show()
```

### InteractiveSemanticExplorer

#### 初始化
```python
explorer = InteractiveSemanticExplorer(
    embeddings=embeddings,           # 嵌入向量矩阵
    metadata=metadata_df            # 元数据DataFrame（可选）
)
```

#### 主要方法

##### 1. project_3d()
创建3D语义空间投影。

```python
# 基本投影
projection_fig = explorer.project_3d(method='UMAP')

# 按元数据列着色
projection_fig = explorer.project_3d(
    method='UMAP',
    color_by='cluster'    # 元数据列名
)
projection_fig.show()
```

##### 2. semantic_search()
执行语义搜索。

```python
# 使用查询嵌入向量搜索
results = explorer.semantic_search(
    query_embedding=query_embedding,
    top_k=10,
    threshold=0.5
)

# 结果包含相似度和元数据
print(results[['document_title', 'similarity', 'journal']])
```

##### 3. find_similar_documents()
查找相似文档。

```python
# 查找与指定文档相似的文档
similar_docs = explorer.find_similar_documents(
    document_index=0,    # 文档索引
    top_k=5
)
```

##### 4. create_similarity_network()
为选定文档创建相似性网络。

```python
# 创建网络
network_html = explorer.create_similarity_network(
    document_indices=[0, 1, 2, 3, 4],  # 文档索引列表
    threshold=0.6                    # 相似性阈值
)
```

### 可视化函数

#### 1. compare_dimensionality_reduction_methods()
对比多种降维方法。

```python
from literature_analyzer import compare_dimensionality_reduction_methods

comparison_fig = compare_dimensionality_reduction_methods(
    embeddings=embeddings,
    methods=['UMAP', 't-SNE', 'PCA'],
    labels=document_titles,
    colors=cluster_labels
)
comparison_fig.show()
```

#### 2. create_multi_dimensional_semantic_space()
创建多维度语义空间对比图。

```python
from literature_analyzer import create_multi_dimensional_semantic_space

fig = create_multi_dimensional_semantic_space(
    embeddings=embeddings,
    labels=document_titles,
    colors=cluster_labels,
    methods=['UMAP', 't-SNE', 'PCA']
)
fig.show()
```

#### 3. create_semantic_space_3d_interactive()
创建交互式3D语义空间。

```python
from literature_analyzer import create_semantic_space_3d_interactive

fig = create_semantic_space_3d_interactive(
    embeddings=embeddings,
    labels=document_titles,
    colors=cluster_labels,
    method='UMAP'
)
fig.show()
```

#### 4. create_semantic_clustering_analysis()
创建语义聚类分析图。

```python
from literature_analyzer import create_semantic_clustering_analysis

fig = create_semantic_clustering_analysis(
    embeddings=embeddings,
    labels=document_titles,
    n_clusters=5
)
fig.show()
```

#### 5. create_semantic_drift_analysis()
创建语义漂移分析图。

```python
from literature_analyzer import create_semantic_drift_analysis

fig = create_semantic_drift_analysis(
    embeddings_1=embeddings_2020,
    embeddings_2=embeddings_2023,
    labels_1=titles_2020,
    labels_2=titles_2023
)
fig.show()
```

## 示例代码

### 完整示例：分析文献语义空间

```python
import numpy as np
import pandas as pd
from literature_analyzer import (
    SemanticSpaceAnalyzer,
    create_multi_dimensional_semantic_space,
    create_semantic_clustering_analysis
)

# 1. 准备数据（假设已有嵌入向量和元数据）
embeddings = np.load('document_embeddings.npy')
metadata = pd.read_csv('document_metadata.csv')

# 2. 创建分析器
analyzer = SemanticSpaceAnalyzer(
    embeddings=embeddings,
    labels=metadata['title'].tolist()
)

# 3. 计算相似性矩阵
similarity_matrix = analyzer.compute_similarity_matrix()
print(f"相似性矩阵形状: {similarity_matrix.shape}")

# 4. 降维分析
reduced_embeddings = analyzer.reduce_dimensions(method='UMAP')
print(f"降维后形状: {reduced_embeddings.shape}")

# 5. 聚类分析
cluster_labels = analyzer.perform_clustering(method='KMeans')
print(f"发现 {len(set(cluster_labels))} 个聚类")

# 6. 获取统计信息
stats = analyzer.get_semantic_statistics()
print("语义空间统计:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# 7. 创建可视化
# 多维度降维对比
comparison_fig = create_multi_dimensional_semantic_space(
    embeddings=embeddings,
    labels=metadata['title'].tolist(),
    colors=cluster_labels,
    methods=['UMAP', 't-SNE', 'PCA']
)
comparison_fig.write_html("dimensionality_reduction_comparison.html")

# 聚类分析图
clustering_fig = create_semantic_clustering_analysis(
    embeddings=embeddings,
    labels=metadata['title'].tolist(),
    n_clusters=len(set(cluster_labels))
)
clustering_fig.write_html("clustering_analysis.html")

print("分析完成！可视化已保存。")
```

### 时间序列分析示例

```python
from literature_analyzer import TemporalSemanticAnalyzer
import pandas as pd

# 1. 创建时间序列分析器
temporal_analyzer = TemporalSemanticAnalyzer()

# 2. 添加多个年份的数据
years = [2018, 2019, 2020, 2021, 2022, 2023]
for year in years:
    # 加载该年份的嵌入向量
    embeddings = np.load(f'embeddings_{year}.npy')
    metadata = pd.read_csv(f'metadata_{year}.csv')
    
    temporal_analyzer.add_time_step(
        time_step=year,
        embeddings=embeddings,
        labels=metadata['title'].tolist()
    )

# 3. 分析演化
evolution_results = temporal_analyzer.analyze_temporal_evolution()
print("演化分析结果:")
print(f"时间步: {evolution_results['time_steps']}")
print(f"平均漂移: {evolution_results['stability_metrics']['mean_drift']:.3f}")
print(f"漂移趋势: {evolution_results['stability_metrics']['drift_trend']}")

# 4. 创建演化动画
animation_fig = temporal_analyzer.create_evolution_animation()
animation_fig.write_html("semantic_evolution_animation.html")

print("时间序列分析完成！")
```

## 最佳实践

### 1. 数据预处理
- 确保嵌入向量已经标准化
- 处理缺失值和异常值
- 考虑使用PCA进行初步降维以减少噪声

### 2. 参数选择
- **UMAP参数**：
  - `n_neighbors`: 通常在5-50之间，数据量大时增大
  - `min_dist`: 控制聚类紧密程度，通常0.01-0.5
  - `metric`: 根据数据特性选择（余弦相似度适合文本）

- **聚类参数**：
  - 使用轮廓系数评估聚类质量
  - 对于DBSCAN，通过k-distance图选择eps
  - 考虑领域知识确定合理的聚类数

### 3. 性能优化
- 对于大数据集（>10,000样本）：
  - 使用采样进行初步探索
  - 考虑使用Dask进行并行计算
  - 启用GPU加速（如果可用）

- 内存管理：
  - 及时删除不再需要的大矩阵
  - 使用稀疏矩阵存储相似性
  - 分批处理时间序列数据

### 4. 可视化技巧
- 使用交互式图表探索数据
- 添加适当的悬停信息
- 使用颜色编码展示类别信息
- 考虑3D旋转动画展示空间结构

### 5. 结果解释
- 结合领域知识解释聚类结果
- 关注异常点和边界情况
- 验证语义漂移的实际意义
- 考虑使用SHAP值解释模型决策

## 常见问题

### Q1: 如何选择合适的降维方法？
**A**: 
- **UMAP**: 适合保留局部和全局结构，计算效率高，推荐作为首选
- **t-SNE**: 适合可视化局部结构，但计算成本高，参数敏感
- **PCA**: 线性方法，计算快速，适合初步探索和降维

建议先用`compare_dimensionality_reduction_methods()`对比效果，根据数据特性选择。

### Q2: 聚类数量如何确定？
**A**: 
- 使用肘部法则观察惯性下降
- 计算轮廓系数，选择使系数最大的k值
- 考虑Calinski-Harabasz指数和Davies-Bouldin指数
- 结合领域知识和实际需求

`SemanticSpaceAnalyzer`的`perform_clustering()`方法在未指定聚类数时会自动选择最优值。

### Q3: 如何处理大规模数据集？
**A**: 
- 使用采样进行初步分析（建议1000-5000个样本）
- 启用`numba`加速计算
- 考虑使用近似最近邻算法
- 分批处理时间序列数据
- 使用Dask进行分布式计算

### Q4: 语义漂移指标如何解释？
**A**: 
- **质心漂移**: 衡量语义空间中心的移动距离，值越大表示变化越大
- **分布漂移**: 使用Wasserstein距离衡量分布变化，考虑各维度差异
- **成对漂移**: 衡量对应文档间距离的变化，适合有对应关系的数据

结合时间背景和领域事件解释漂移的实际意义。

### Q5: 网络图太密集怎么办？
**A**: 
- 提高相似性阈值（如从0.7提高到0.8）
- 限制最大节点数量
- 使用力导向布局参数调整节点分布
- 考虑只显示高权重边
- 使用聚类简化网络结构

### Q6: 如何保存和分享结果？
**A**: 
- 使用`fig.write_html()`保存交互式图表
- 网络图可以保存为HTML文件直接在浏览器中查看
- 使用`plt.savefig()`保存静态图片
- 考虑创建Jupyter Notebook展示完整分析流程
- 重要结果可以导出为CSV或Excel文件

## 进阶主题

### 1. 自定义降维方法
可以通过继承现有类或实现新接口来添加自定义降维方法：

```python
from sklearn.decomposition import TruncatedSVD

class CustomSemanticAnalyzer(SemanticSpaceAnalyzer):
    def reduce_dimensions(self, method='Custom', **kwargs):
        if method == 'Custom':
            reducer = TruncatedSVD(n_components=kwargs.get('n_components', 2))
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        else:
            return super().reduce_dimensions(method, **kwargs)
        return self.reduced_embeddings
```

### 2. 集成外部工具
可以集成以下工具增强分析能力：
- **Gephi**: 复杂网络分析和可视化
- **Tableau**: 交互式仪表板
- **R**: 高级统计分析和特定算法
- **Neo4j**: 图数据库存储和查询

### 3. 实时分析
对于需要实时更新的场景：
- 使用Streamlit创建实时仪表板
- 设置定时任务更新数据
- 使用WebSocket推送更新
- 实现增量计算避免全量重算

### 4. 多语言支持
虽然当前主要支持中文和英文，但可以通过：
- 添加多语言文本预处理
- 使用多语言嵌入模型
- 实现语言检测和切换
来扩展多语言支持。

## 总结

高级语义空间分析功能为文献分析器提供了强大的语义理解和可视化能力。通过合理使用这些工具，可以：

1. 深入理解文献集合的语义结构
2. 发现文献间的隐藏关系
3. 跟踪研究领域的演化趋势
4. 识别研究热点和新兴方向
5. 提供直观的可视化展示

建议从简单示例开始，逐步探索更复杂的功能，并结合具体研究问题调整参数和方法。

如需更多帮助，请参考示例代码或联系开发团队。
