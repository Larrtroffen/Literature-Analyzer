"""
语义空间分析模块

提供高级语义空间分析和可视化功能，包括：
- 多维度降维对比
- 语义相似性网络
- 语义演化分析
- 语义聚类和漂移检测
- 交互式3D可视化
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict

# 配置日志
logger = logging.getLogger(__name__)


class SemanticSpaceAnalyzer:
    """语义空间分析器类"""
    
    def __init__(self, embeddings: np.ndarray, labels: Optional[List[str]] = None):
        """
        初始化语义空间分析器。
        
        Args:
            embeddings: 嵌入向量矩阵
            labels: 标签列表（可选）
        """
        self.embeddings = embeddings
        self.labels = labels
        self.reduced_embeddings = None
        self.cluster_labels = None
        self.similarity_matrix = None
        
        logger.info(f"语义空间分析器初始化完成，嵌入向量形状: {embeddings.shape}")
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        计算嵌入向量的余弦相似性矩阵。
        
        Returns:
            相似性矩阵
        """
        logger.info("开始计算相似性矩阵...")
        
        self.similarity_matrix = cosine_similarity(self.embeddings)
        
        logger.info(f"相似性矩阵计算完成，形状: {self.similarity_matrix.shape}")
        
        return self.similarity_matrix
    
    def reduce_dimensions(
        self,
        method: str = 'UMAP',
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        对嵌入向量进行降维。
        
        Args:
            method: 降维方法 ('UMAP', 't-SNE', 'PCA')
            n_components: 目标维度
            **kwargs: 降维方法的额外参数
            
        Returns:
            降维后的嵌入向量
        """
        logger.info(f"开始使用{method}进行降维...")
        
        if method == 'UMAP':
            reducer = umap.UMAP(n_components=n_components, **kwargs)
        elif method == 't-SNE':
            reducer = TSNE(n_components=n_components, **kwargs)
        elif method == 'PCA':
            reducer = PCA(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        logger.info(f"降维完成，结果形状: {self.reduced_embeddings.shape}")
        
        return self.reduced_embeddings
    
    def perform_clustering(
        self,
        method: str = 'KMeans',
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        对嵌入向量进行聚类分析。
        
        Args:
            method: 聚类方法 ('KMeans', 'DBSCAN', 'Agglomerative')
            n_clusters: 聚类数量（可选，自动确定）
            **kwargs: 聚类方法的额外参数
            
        Returns:
            聚类标签
        """
        logger.info(f"开始使用{method}进行聚类分析...")
        
        if method == 'KMeans':
            if n_clusters is None:
                n_clusters = self._optimal_kmeans_clusters()
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
        elif method == 'DBSCAN':
            clusterer = DBSCAN(**kwargs)
        elif method == 'Agglomerative':
            if n_clusters is None:
                n_clusters = self._optimal_kmeans_clusters()
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        self.cluster_labels = clusterer.fit_predict(self.embeddings)
        
        # 计算聚类评估指标
        if len(set(self.cluster_labels)) > 1:
            silhouette = silhouette_score(self.embeddings, self.cluster_labels)
            calinski = calinski_harabasz_score(self.embeddings, self.cluster_labels)
            davies = davies_bouldin_score(self.embeddings, self.cluster_labels)
            
            logger.info(f"聚类完成，聚类数: {len(set(self.cluster_labels))}")
            logger.info(f"轮廓系数: {silhouette:.3f}")
            logger.info(f"Calinski-Harabasz指数: {calinski:.3f}")
            logger.info(f"Davies-Bouldin指数: {davies:.3f}")
        else:
            logger.warning("聚类结果只有一个簇，无法计算评估指标")
        
        return self.cluster_labels
    
    def _optimal_kmeans_clusters(self, max_clusters: int = 10) -> int:
        """
        使用肘部法则确定最优KMeans聚类数。
        
        Args:
            max_clusters: 最大聚类数
            
        Returns:
            最优聚类数
        """
        if len(self.embeddings) < max_clusters * 2:
            return max(2, len(self.embeddings) // 10)
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, min(max_clusters + 1, len(self.embeddings) // 2)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            inertias.append(kmeans.inertia_)
            
            if len(set(cluster_labels)) > 1:
                silhouette_scores.append(silhouette_score(self.embeddings, cluster_labels))
            else:
                silhouette_scores.append(0)
        
        # 选择轮廓系数最大的k值
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        logger.info(f"最优聚类数: {optimal_k}")
        
        return optimal_k
    
    def analyze_semantic_drift(
        self,
        other_embeddings: np.ndarray,
        method: str = 'centroid'
    ) -> Dict[str, float]:
        """
        分析语义漂移。
        
        Args:
            other_embeddings: 另一个时间点的嵌入向量
            method: 分析方法 ('centroid', 'distribution', 'pairwise')
            
        Returns:
            语义漂移分析结果
        """
        logger.info(f"开始使用{method}方法分析语义漂移...")
        
        drift_metrics = {}
        
        if method == 'centroid':
            # 质心漂移
            centroid1 = np.mean(self.embeddings, axis=0)
            centroid2 = np.mean(other_embeddings, axis=0)
            drift_distance = np.linalg.norm(centroid2 - centroid1)
            drift_metrics['centroid_drift'] = drift_distance
            
        elif method == 'distribution':
            # 分布漂移（Wasserstein距离）
            try:
                from scipy.stats import wasserstein_distance
                
                # 计算每个维度的Wasserstein距离
                dim_distances = []
                for dim in range(self.embeddings.shape[1]):
                    dist = wasserstein_distance(self.embeddings[:, dim], other_embeddings[:, dim])
                    dim_distances.append(dist)
                
                drift_metrics['distribution_drift'] = np.mean(dim_distances)
                drift_metrics['distribution_drift_std'] = np.std(dim_distances)
                
            except ImportError:
                logger.warning("scipy未安装，无法计算Wasserstein距离")
        
        elif method == 'pairwise':
            # 成对距离变化
            if len(self.embeddings) == len(other_embeddings):
                pairwise_distances = cdist(self.embeddings, other_embeddings, metric='cosine')
                drift_metrics['avg_pairwise_drift'] = np.mean(np.diag(pairwise_distances))
                drift_metrics['max_pairwise_drift'] = np.max(np.diag(pairwise_distances))
            else:
                logger.warning("嵌入向量数量不匹配，无法计算成对距离")
        
        logger.info(f"语义漂移分析完成: {drift_metrics}")
        
        return drift_metrics
    
    def create_semantic_network(
        self,
        threshold: float = 0.7,
        max_nodes: int = 100
    ) -> str:
        """
        创建语义相似性网络图。
        
        Args:
            threshold: 相似性阈值
            max_nodes: 最大节点数量
            
        Returns:
            HTML字符串（pyvis网络图）
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        logger.info("开始创建语义相似性网络...")
        
        # 采样
        sample_size = min(max_nodes, len(self.embeddings))
        if len(self.embeddings) > sample_size:
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            sampled_similarity = self.similarity_matrix[np.ix_(indices, indices)]
            sampled_labels = [self.labels[i] for i in indices] if self.labels else [f"Doc_{i}" for i in indices]
        else:
            sampled_similarity = self.similarity_matrix
            sampled_labels = self.labels if self.labels else [f"Doc_{i}" for i in range(len(self.embeddings))]
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for i, label in enumerate(sampled_labels):
            G.add_node(i, label=label, title=f"文档: {label}")
        
        # 添加边（基于相似性阈值）
        edge_count = 0
        for i in range(len(sampled_similarity)):
            for j in range(i + 1, len(sampled_similarity)):
                similarity = sampled_similarity[i, j]
                if similarity >= threshold:
                    G.add_edge(i, j, weight=similarity, title=f"相似度: {similarity:.3f}")
                    edge_count += 1
        
        # 使用pyvis创建交互式网络
        net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
        net.from_nx(G)
        
        # 设置节点样式
        for node in net.nodes:
            node['size'] = 15
            node['borderWidth'] = 2
            node['color'] = '#4CAF50'
        
        # 设置边样式
        for edge in net.edges:
            edge['width'] = max(1, edge['weight'] * 5)
            edge['color'] = '#FFC107'
        
        # 设置物理布局
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -50000,
              "springConstant": 0.01,
              "damping": 0.2
            },
            "minVelocity": 0.75
          }
        }
        """)
        
        # 生成HTML
        html_path = "semantic_network.html"
        net.save_graph(html_path)
        
        # 读取HTML内容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 清理临时文件
        import os
        os.remove(html_path)
        
        logger.info(f"语义相似性网络创建完成，包含{len(G.nodes())}个节点，{edge_count}条边")
        
        return html_content
    
    def create_density_heatmap(
        self,
        grid_size: int = 50
    ) -> go.Figure:
        """
        创建语义密度热力图。
        
        Args:
            grid_size: 网格大小
            
        Returns:
            Plotly热力图对象
        """
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        logger.info("开始创建语义密度热力图...")
        
        reduced = self.reduced_embeddings
        
        # 创建网格
        x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
        y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # 计算密度
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([reduced[:, 0], reduced[:, 1]])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            x=x_grid,
            y=y_grid,
            z=Z,
            colorscale='Viridis',
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>密度: %{z:.3f}<extra></extra>'
        ))
        
        # 添加原始数据点
        fig.add_trace(go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode='markers',
            marker=dict(
                size=4,
                color='white',
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=self.labels if self.labels else None,
            hovertemplate='%{text}<extra></extra>' if self.labels else '<extra></extra>',
            name='数据点'
        ))
        
        # 更新布局
        fig.update_layout(
            title="语义密度热力图",
            xaxis_title="维度1",
            yaxis_title="维度2",
            height=600,
            showlegend=False
        )
        
        logger.info(f"语义密度热力图创建完成，网格大小: {grid_size}x{grid_size}")
        
        return fig
    
    def get_semantic_statistics(self) -> Dict[str, Any]:
        """
        获取语义空间统计信息。
        
        Returns:
            统计信息字典
        """
        logger.info("开始计算语义空间统计信息...")
        
        stats = {}
        
        # 基本统计
        stats['embedding_shape'] = self.embeddings.shape
        stats['embedding_dimension'] = self.embeddings.shape[1]
        stats['num_documents'] = self.embeddings.shape[0]
        
        # 计算相似性统计
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # 只取上三角部分（不包括对角线）
        upper_triangular = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        
        stats['mean_similarity'] = np.mean(upper_triangular)
        stats['std_similarity'] = np.std(upper_triangular)
        stats['min_similarity'] = np.min(upper_triangular)
        stats['max_similarity'] = np.max(upper_triangular)
        
        # 计算嵌入向量的统计
        stats['embedding_norm_mean'] = np.mean(np.linalg.norm(self.embeddings, axis=1))
        stats['embedding_norm_std'] = np.std(np.linalg.norm(self.embeddings, axis=1))
        
        # 聚类统计（如果已执行）
        if self.cluster_labels is not None:
            unique_labels = set(self.cluster_labels)
            stats['num_clusters'] = len(unique_labels)
            stats['cluster_sizes'] = [np.sum(self.cluster_labels == label) for label in unique_labels]
            
            if len(unique_labels) > 1:
                stats['silhouette_score'] = silhouette_score(self.embeddings, self.cluster_labels)
                stats['calinski_harabasz_score'] = calinski_harabasz_score(self.embeddings, self.cluster_labels)
                stats['davies_bouldin_score'] = davies_bouldin_score(self.embeddings, self.cluster_labels)
        
        logger.info("语义空间统计信息计算完成")
        
        return stats


class TemporalSemanticAnalyzer:
    """时间序列语义分析器类"""
    
    def __init__(self):
        """初始化时间序列语义分析器"""
        self.temporal_embeddings = {}
        self.temporal_labels = {}
        self.drift_history = []
        
        logger.info("时间序列语义分析器初始化完成")
    
    def add_time_step(
        self,
        time_step: int,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> None:
        """
        添加时间步数据。
        
        Args:
            time_step: 时间步
            embeddings: 嵌入向量
            labels: 标签（可选）
        """
        self.temporal_embeddings[time_step] = embeddings
        if labels is not None:
            self.temporal_labels[time_step] = labels
        
        logger.info(f"添加时间步 {time_step}，嵌入向量形状: {embeddings.shape}")
    
    def analyze_temporal_evolution(self) -> Dict[str, Any]:
        """
        分析时间演化。
        
        Returns:
            时间演化分析结果
        """
        logger.info("开始分析时间演化...")
        
        if len(self.temporal_embeddings) < 2:
            logger.warning("需要至少2个时间步才能分析时间演化")
            return {}
        
        evolution_results = {
            'time_steps': sorted(self.temporal_embeddings.keys()),
            'drift_metrics': {},
            'topic_evolution': {},
            'stability_metrics': {}
        }
        
        # 分析每个时间步之间的漂移
        time_steps = sorted(self.temporal_embeddings.keys())
        for i in range(len(time_steps) - 1):
            current_step = time_steps[i]
            next_step = time_steps[i + 1]
            
            current_embeddings = self.temporal_embeddings[current_step]
            next_embeddings = self.temporal_embeddings[next_step]
            
            # 创建分析器并计算漂移
            analyzer = SemanticSpaceAnalyzer(current_embeddings)
            drift_metrics = analyzer.analyze_semantic_drift(next_embeddings)
            
            evolution_results['drift_metrics'][f"{current_step}_{next_step}"] = drift_metrics
            self.drift_history.append(drift_metrics)
        
        # 计算稳定性指标
        if self.drift_history:
            drift_distances = [metrics.get('centroid_drift', 0) for metrics in self.drift_history]
            evolution_results['stability_metrics']['mean_drift'] = np.mean(drift_distances)
            evolution_results['stability_metrics']['drift_trend'] = 'increasing' if drift_distances[-1] > drift_distances[0] else 'decreasing'
            evolution_results['stability_metrics']['drift_volatility'] = np.std(drift_distances)
        
        logger.info("时间演化分析完成")
        
        return evolution_results
    
    def create_evolution_animation(self) -> go.Figure:
        """
        创建语义空间演化动画。
        
        Returns:
            Plotly动画图对象
        """
        if len(self.temporal_embeddings) < 2:
            logger.warning("需要至少2个时间步才能创建演化动画")
            return go.Figure()
        
        logger.info("开始创建语义空间演化动画...")
        
        frames = []
        time_steps = sorted(self.temporal_embeddings.keys())
        
        # 为每个时间步创建帧
        for step in time_steps:
            embeddings = self.temporal_embeddings[step]
            step_labels = self.temporal_labels.get(step, None)
            
            # 降维到2D
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            
            # 创建帧数据
            frame_data = go.Scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightblue',
                    opacity=0.7
                ),
                text=step_labels if step_labels else None,
                hovertemplate='%{text}<extra></extra>' if step_labels else '<extra></extra>',
                name=f'时间步 {step}'
            )
            
            frames.append(go.Frame(data=[frame_data], name=f"frame_{step}"))
        
        # 创建初始图（使用第一个时间步）
        initial_step = time_steps[0]
        initial_embeddings = self.temporal_embeddings[initial_step]
        initial_labels = self.temporal_labels.get(initial_step, None)
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        initial_reduced = reducer.fit_transform(initial_embeddings)
        
        fig = go.Figure(
            data=[go.Scatter(
                x=initial_reduced[:, 0],
                y=initial_reduced[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightblue',
                    opacity=0.7
                ),
                text=initial_labels if initial_labels else None,
                hovertemplate='%{text}<extra></extra>' if initial_labels else '<extra></extra>',
                name=f'时间步 {initial_step}'
            )],
            frames=frames
        )
        
        # 添加动画控制
        fig.update_layout(
            title="语义空间演化动画",
            xaxis_title="维度1",
            yaxis_title="维度2",
            height=600,
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "播放",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "暂停",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ]
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[f"frame_{step}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": f"时间步 {step}",
                        "method": "animate"
                    }
                    for step in time_steps
                ]
            }]
        )
        
        logger.info(f"语义空间演化动画创建完成，包含{len(time_steps)}个时间步")
        
        return fig


class InteractiveSemanticExplorer:
    """交互式语义空间探索器类"""
    
    def __init__(self, embeddings: np.ndarray, metadata: Optional[pd.DataFrame] = None):
        """
        初始化交互式语义空间探索器。
        
        Args:
            embeddings: 嵌入向量矩阵
            metadata: 元数据DataFrame（可选）
        """
        self.embeddings = embeddings
        self.metadata = metadata
        self.current_projection = None
        self.selected_points = []
        self.search_history = []
        
        logger.info(f"交互式语义空间探索器初始化完成，嵌入向量形状: {embeddings.shape}")
    
    def project_3d(
        self,
        method: str = 'UMAP',
        color_by: Optional[str] = None
    ) -> go.Figure:
        """
        创建3D语义空间投影。
        
        Args:
            method: 降维方法
            color_by: 颜色映射列名
            
        Returns:
            Plotly 3D散点图对象
        """
        logger.info(f"开始创建3D语义空间投影，使用{method}降维...")
        
        # 应用3D降维
        if method == 'UMAP':
            reducer = umap.UMAP(n_components=3, random_state=42)
        elif method == 't-SNE':
            reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(self.embeddings)-1))
        elif method == 'PCA':
            reducer = PCA(n_components=3, random_state=42)
        else:
            logger.warning(f"不支持的降维方法: {method}，使用UMAP")
            reducer = umap.UMAP(n_components=3, random_state=42)
        
        reduced_3d = reducer.fit_transform(self.embeddings)
        self.current_projection = reduced_3d
        
        # 确定颜色
        colors = None
        if color_by is not None and self.metadata is not None and color_by in self.metadata.columns:
            colors = self.metadata[color_by].values
        
        # 创建3D散点图
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_3d[:, 0],
            y=reduced_3d[:, 1],
            z=reduced_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors if colors is not None else 'lightblue',
                opacity=0.8,
                colorscale='Viridis',
                showscale=True if colors is not None else False,
                colorbar=dict(title=color_by if colors is not None else None)
            ),
            text=self.metadata['article_title'].values if self.metadata is not None and 'article_title' in self.metadata.columns else None,
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<extra></extra>' if self.metadata is not None and 'article_title' in self.metadata.columns else
                         'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            name='语义空间'
        )])
        
        # 更新布局
        fig.update_layout(
            title=f"交互式3D语义空间 ({method})",
            scene=dict(
                xaxis_title="维度1",
                yaxis_title="维度2",
                zaxis_title="维度3",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700
        )
        
        # 添加控制按钮
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.5}}],
                            label="默认视角",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5}}],
                            label="俯视图",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0}}],
                            label="侧视图",
                            method="relayout"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        logger.info(f"3D语义空间投影创建完成，使用{method}降维")
        
        return fig
    
    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        执行语义搜索。
        
        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回结果数量
            threshold: 相似性阈值
            
        Returns:
            搜索结果DataFrame
        """
        logger.info(f"开始语义搜索，top_k={top_k}, threshold={threshold}")
        
        # 计算相似性
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 筛选结果
        valid_indices = np.where(similarities >= threshold)[0]
        valid_similarities = similarities[valid_indices]
        
        # 排序并取top_k
        top_indices = valid_indices[np.argsort(valid_similarities)[-top_k:][::-1]]
        top_similarities = similarities[top_indices]
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'index': top_indices,
            'similarity': top_similarities
        })
        
        # 添加元数据
        if self.metadata is not None:
            for col in self.metadata.columns:
                if col in self.metadata.columns:
                    results[col] = self.metadata.iloc[top_indices][col].values
        
        # 记录搜索历史
        self.search_history.append({
            'query': query_embedding,
            'results': results.copy(),
            'timestamp': pd.Timestamp.now()
        })
        
        logger.info(f"语义搜索完成，找到{len(results)}个结果")
        
        return results
    
    def find_similar_documents(
        self,
        document_index: int,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        查找相似文档。
        
        Args:
            document_index: 文档索引
            top_k: 返回结果数量
            
        Returns:
            相似文档DataFrame
        """
        if document_index >= len(self.embeddings):
            raise ValueError(f"文档索引 {document_index} 超出范围")
        
        query_embedding = self.embeddings[document_index]
        return self.semantic_search(query_embedding, top_k)
    
    def create_similarity_network(
        self,
        document_indices: List[int],
        threshold: float = 0.6
    ) -> str:
        """
        为选定文档创建相似性网络。
        
        Args:
            document_indices: 文档索引列表
            threshold: 相似性阈值
            
        Returns:
            HTML字符串（pyvis网络图）
        """
        logger.info(f"开始为选定文档创建相似性网络，文档数量: {len(document_indices)}")
        
        # 提取选定文档的嵌入向量
        selected_embeddings = self.embeddings[document_indices]
        
        # 计算相似性矩阵
        similarity_matrix = cosine_similarity(selected_embeddings)
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for i, idx in enumerate(document_indices):
            label = f"Doc_{idx}"
            if self.metadata is not None and 'article_title' in self.metadata.columns:
                title = self.metadata.iloc[idx]['article_title'][:50] + "..."
            else:
                title = label
            
            G.add_node(i, label=label, title=title, document_index=idx)
        
        # 添加边
        edge_count = 0
        for i in range(len(document_indices)):
            for j in range(i + 1, len(document_indices)):
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    G.add_edge(i, j, weight=similarity, title=f"相似度: {similarity:.3f}")
                    edge_count += 1
        
        # 使用pyvis创建交互式网络
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
        net.from_nx(G)
        
        # 设置节点样式
        for node in net.nodes:
            node['size'] = 20
            node['borderWidth'] = 2
            node['color'] = '#2196F3'
        
        # 设置边样式
        for edge in net.edges:
            edge['width'] = max(2, edge['weight'] * 8)
            edge['color'] = '#FF9800'
        
        # 设置物理布局
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -30000,
              "springConstant": 0.02,
              "damping": 0.15
            },
            "minVelocity": 0.75
          }
        }
        """)
        
        # 生成HTML
        html_path = "document_similarity_network.html"
        net.save_graph(html_path)
        
        # 读取HTML内容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 清理临时文件
        import os
        os.remove(html_path)
        
        logger.info(f"文档相似性网络创建完成，包含{len(G.nodes())}个节点，{edge_count}条边")
        
        return html_content


# 工具函数

def compare_dimensionality_reduction_methods(
    embeddings: np.ndarray,
    methods: List[str] = ['UMAP', 't-SNE', 'PCA'],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None
) -> go.Figure:
    """
    比较多种降维方法。
    
    Args:
        embeddings: 嵌入向量矩阵
        methods: 降维方法列表
        labels: 标签列表（可选）
        colors: 颜色列表（可选）
        
    Returns:
        Plotly子图对象
    """
    logger.info(f"开始比较降维方法: {methods}")
    
    # 采样（如果数据量太大）
    sample_size = min(2000, len(embeddings))
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sampled_embeddings = embeddings[indices]
        sampled_labels = [labels[i] for i in indices] if labels else None
        sampled_colors = [colors[i] for i in indices] if colors else None
    else:
        sampled_embeddings = embeddings
        sampled_labels = labels
        sampled_colors = colors
    
    # 创建子图布局
    n_methods = len(methods)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=methods,
        specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
    )
    
    # 对每种降维方法进行处理
    for i, method in enumerate(methods):
        row = i // cols + 1
        col = i % cols + 1
        
        # 应用降维方法
        if method == 'UMAP':
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(sampled_embeddings)
        elif method == 't-SNE':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sampled_embeddings)-1))
            reduced = reducer.fit_transform(sampled_embeddings)
        elif method == 'PCA':
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(sampled_embeddings)
        else:
            logger.warning(f"不支持的降维方法: {method}")
            continue
        
        # 创建散点图
        scatter = go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=sampled_colors if sampled_colors else 'lightblue',
                opacity=0.7,
                showscale=True if sampled_colors else False
            ),
            text=sampled_labels if sampled_labels else None,
            hovertemplate='%{text}<extra></extra>' if sampled_labels else '<extra></extra>',
            name=method
        )
        
        fig.add_trace(scatter, row=row, col=col)
    
    # 更新布局
    fig.update_layout(
        title="多维度语义空间对比",
        height=400 * rows,
        showlegend=False,
        hovermode='closest'
    )
    
    # 统一坐标轴标签
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(title_text="维度1", row=i, col=j)
            fig.update_yaxes(title_text="维度2", row=i, col=j)
    
    logger.info(f"降维方法对比图创建完成，包含 {len(methods)} 种方法")
    
    return fig


def create_semantic_drift_comparison(
    embeddings_dict: Dict[str, np.ndarray],
    reference_key: str = 'first'
) -> go.Figure:
    """
    创建多个语义空间之间的漂移对比图。
    
    Args:
        embeddings_dict: 嵌入向量字典
        reference_key: 参考键
        
    Returns:
        Plotly漂移对比图对象
    """
    logger.info("开始创建语义空间漂移对比图...")
    
    if reference_key not in embeddings_dict:
        logger.warning(f"参考键 '{reference_key}' 不存在于嵌入向量字典中")
        return go.Figure()
    
    reference_embeddings = embeddings_dict[reference_key]
    other_keys = [k for k in embeddings_dict.keys() if k != reference_key]
    
    if not other_keys:
        logger.warning("没有其他嵌入向量用于比较")
        return go.Figure()
    
    # 创建子图布局
    n_others = len(other_keys)
    cols = min(3, n_others)
    rows = (n_others + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{reference_key} vs {key}" for key in other_keys],
        specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
    )
    
    # 对每个其他嵌入向量进行比较
    for i, key in enumerate(other_keys):
        row = i // cols + 1
        col = i % cols + 1
        
        other_embeddings = embeddings_dict[key]
        
        # 创建分析器并计算漂移
        analyzer = SemanticSpaceAnalyzer(reference_embeddings)
        drift_metrics = analyzer.analyze_semantic_drift(other_embeddings)
        
        # 合并数据进行降维
        combined_embeddings = np.vstack([reference_embeddings, other_embeddings])
        reducer = umap.UMAP(n_components=2, random_state=42)
        combined_reduced = reducer.fit_transform(combined_embeddings)
        
        # 分离降维结果
        reduced_ref = combined_reduced[:len(reference_embeddings)]
        reduced_other = combined_reduced[len(reference_embeddings):]
        
        # 绘制参考数据
        fig.add_trace(go.Scatter(
            x=reduced_ref[:, 0],
            y=reduced_ref[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='lightblue',
                opacity=0.7
            ),
            name=f'{reference_key}',
            showlegend=False
        ), row=row, col=col)
        
        # 绘制其他数据
        fig.add_trace(go.Scatter(
            x=reduced_other[:, 0],
            y=reduced_other[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='lightcoral',
                opacity=0.7
            ),
            name=f'{key}',
            showlegend=False
        ), row=row, col=col)
        
        # 添加漂移信息注释
        drift_text = f"质心漂移: {drift_metrics.get('centroid_drift', 0):.3f}"
        fig.add_annotation(
            text=drift_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
            row=row, col=col
        )
    
    # 更新布局
    fig.update_layout(
        title="语义空间漂移对比",
        height=500 * rows,
        showlegend=False
    )
    
    # 统一坐标轴
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(title_text="维度1", row=i, col=j)
            fig.update_yaxes(title_text="维度2", row=i, col=j)
    
    logger.info(f"语义空间漂移对比图创建完成，比较了 {len(other_keys)} 个空间")
    
    return fig
