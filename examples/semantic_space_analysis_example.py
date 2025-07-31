"""
语义空间分析示例脚本

展示如何使用新添加的高级语义空间分析功能，包括：
- 多维度降维对比
- 语义相似性网络
- 语义演化分析
- 语义聚类和漂移检测
- 交互式3D可视化
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from literature_analyzer import (
    SemanticSpaceAnalyzer,
    TemporalSemanticAnalyzer,
    InteractiveSemanticExplorer,
    compare_dimensionality_reduction_methods,
    create_semantic_drift_comparison,
    create_multi_dimensional_semantic_space,
    create_semantic_space_3d_interactive,
    create_semantic_similarity_network,
    create_semantic_evolution_animation,
    create_semantic_density_heatmap,
    create_semantic_clustering_analysis,
    create_semantic_drift_analysis,
)
import plotly.graph_objects as go


def generate_sample_data(n_samples=500, n_features=100, n_clusters=5):
    """
    生成示例数据。
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_clusters: 聚类数量
        
    Returns:
        嵌入入向量和标签
    """
    print(f"生成示例数据: {n_samples}个样本, {n_features}维特征, {n_clusters}个聚类")
    
    # 生成聚类数据
    embeddings, labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=42
    )
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.5, embeddings.shape)
    embeddings = embeddings + noise
    
    # 生成文档标题
    document_titles = [f"Document_{i}" for i in range(n_samples)]
    
    # 创建元数据DataFrame
    metadata = pd.DataFrame({
        'document_title': document_titles,
        'cluster': labels,
        'year': np.random.randint(2010, 2023, n_samples),
        'journal': np.random.choice(['Journal A', 'Journal B', 'Journal C'], n_samples)
    })
    
    print(f"示例数据生成完成")
    
    return embeddings, metadata


def example_basic_semantic_analysis():
    """
    基础语义空间分析示例。
    """
    print("\n=== 基础语义空间分析示例 ===")
    
    # 生成示例数据
    embeddings, metadata = generate_sample_data()
    
    # 创建语义空间分析器
    analyzer = SemanticSpaceAnalyzer(embeddings, metadata['document_title'].tolist())
    
    # 计算相似性矩阵
    similarity_matrix = analyzer.compute_similarity_matrix()
    print(f"相似性矩阵形状: {similarity_matrix.shape}")
    
    # 降维
    reduced_embeddings = analyzer.reduce_dimensions(method='UMAP', n_components=2)
    print(f"降维后形状: {reduced_embeddings.shape}")
    
    # 聚类分析
    cluster_labels = analyzer.perform_clustering(method='KMeans')
    print(f"聚类标签数量: {len(set(cluster_labels))}")
    
    # 获取统计信息
    stats = analyzer.get_semantic_statistics()
    print(f"语义空间统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 创建密度热力图
    density_fig = analyzer.create_density_heatmap(grid_size=30)
    density_fig.show()
    
    return analyzer


def example_temporal_semantic_analysis():
    """
    时间序列语义分析示例。
    """
    print("\n=== 时间序列语义分析示例 ===")
    
    # 创建时间序列分析器
    temporal_analyzer = TemporalSemanticAnalyzer()
    
    # 生成多个时间步的数据
    time_steps = [2010, 2015, 2020, 2023]
    for step in time_steps:
        # 为每个时间步生成略有不同的数据
        embeddings, metadata = generate_sample_data(n_samples=200, n_clusters=3)
        
        # 添加时间特定的漂移
        drift = (step - 2010) * 0.1
        embeddings = embeddings + np.random.normal(drift, 0.2, embeddings.shape)
        
        temporal_analyzer.add_time_step(step, embeddings, metadata['document_title'].tolist())
        print(f"添加时间步 {step}: {len(embeddings)}个样本")
    
    # 分析时间演化
    evolution_results = temporal_analyzer.analyze_temporal_evolution()
    print(f"时间演化分析结果:")
    print(f"  时间步: {evolution_results['time_steps']}")
    print(f"  稳定性指标: {evolution_results['stability_metrics']}")
    
    # 创建演化动画
    animation_fig = temporal_analyzer.create_evolution_animation()
    animation_fig.show()
    
    return temporal_analyzer


def example_interactive_semantic_exploration():
    """
    交互式语义空间探索示例。
    """
    print("\n=== 交互式语义空间探索示例 ===")
    
    # 生成示例数据
    embeddings, metadata = generate_sample_data()
    
    # 创建交互式探索器
    explorer = InteractiveSemanticExplorer(embeddings, metadata)
    
    # 创建3D投影
    projection_3d_fig = explorer.project_3d(method='UMAP', color_by='cluster')
    projection_3d_fig.show()
    
    # 语义搜索示例
    query_embedding = embeddings[0]  # 使用第一个文档作为查询
    search_results = explorer.semantic_search(query_embedding, top_k=5)
    print(f"语义搜索结果 (top 5):")
    print(search_results[['document_title', 'similarity']])
    
    # 查找相似文档
    similar_docs = explorer.find_similar_documents(document_index=0, top_k=3)
    print(f"与Document_0相似的文档:")
    print(similar_docs[['document_title', 'similarity']])
    
    # 创建文档相似性网络
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    network_html = explorer.create_similarity_network(selected_indices, threshold=0.6)
    
    # 保存网络图到文件
    with open('semantic_network_example.html', 'w', encoding='utf-8') as f:
        f.write(network_html)
    print("文档相似性网络已保存到 semantic_network_example.html")
    
    return explorer


def example_dimensionality_reduction_comparison():
    """
    降维方法对比示例。
    """
    print("\n=== 降维方法对比示例 ===")
    
    # 生成示例数据
    embeddings, metadata = generate_sample_data(n_features=50)
    
    # 比较多种降维方法
    comparison_fig = compare_dimensionality_reduction_methods(
        embeddings,
        methods=['UMAP', 't-SNE', 'PCA'],
        labels=metadata['document_title'].tolist(),
        colors=metadata['cluster'].tolist()
    )
    comparison_fig.show()
    
    return comparison_fig


def example_semantic_drift_analysis():
    """
    语义漂移分析示例。
    """
    print("\n=== 语义漂移分析示例 ===")
    
    # 生成两个不同时间点的数据
    embeddings_1, metadata_1 = generate_sample_data(n_samples=300, n_clusters=3)
    embeddings_2, metadata_2 = generate_sample_data(n_samples=300, n_clusters=3)
    
    # 添加漂移
    embeddings_2 = embeddings_2 + np.random.normal(1.0, 0.3, embeddings_2.shape)
    
    # 创建分析器并分析漂移
    analyzer = SemanticSpaceAnalyzer(embeddings_1)
    drift_metrics = analyzer.analyze_semantic_drift(embeddings_2)
    print(f"语义漂移分析结果:")
    for key, value in drift_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # 创建漂移对比图
    embeddings_dict = {
        '时间点1': embeddings_1,
        '时间点2': embeddings_2
    }
    drift_fig = create_semantic_drift_comparison(embeddings_dict, reference_key='时间点1')
    drift_fig.show()
    
    return drift_metrics


def example_advanced_clustering():
    """
    高级聚类分析示例。
    """
    print("\n=== 高级聚类分析示例 ===")
    
    # 生成示例数据
    embeddings, metadata = generate_sample_data(n_features=50, n_clusters=4)
    
    # 创建语义空间分析器
    analyzer = SemanticSpaceAnalyzer(embeddings)
    
    # 尝试不同的聚类方法
    clustering_methods = ['KMeans', 'DBSCAN', 'Agglomerative']
    
    for method in clustering_methods:
        print(f"\n使用 {method} 进行聚类:")
        
        if method == 'DBSCAN':
            # DBSCAN需要特殊参数
            cluster_labels = analyzer.perform_clustering(method=method, eps=0.5, min_samples=5)
        else:
            cluster_labels = analyzer.perform_clustering(method=method)
        
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        print(f"  发现的聚类数: {n_clusters}")
        
        if n_clusters > 1:
            silhouette = silhouette_score(embeddings, cluster_labels)
            print(f"  轮廓系数: {silhouette:.3f}")
    
    # 创建聚类分析图
    clustering_fig = create_semantic_clustering_analysis(
        embeddings,
        labels=metadata['document_title'].tolist(),
        n_clusters=4
    )
    clustering_fig.show()
    
    return analyzer


def main():
    """
    主函数，运行所有示例。
    """
    print("开始运行语义空间分析示例...")
    
    try:
        # 基础语义空间分析
        analyzer = example_basic_semantic_analysis()
        
        # 时间序列语义分析
        temporal_analyzer = example_temporal_semantic_analysis()
        
        # 交互式语义空间探索
        explorer = example_interactive_semantic_exploration()
        
        # 降维方法对比
        comparison_fig = example_dimensionality_reduction_comparison()
        
        # 语义漂移分析
        drift_metrics = example_semantic_drift_analysis()
        
        # 高级聚类分析
        cluster_analyzer = example_advanced_clustering()
        
        print("\n所有示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
