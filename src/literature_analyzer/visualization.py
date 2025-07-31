"""
可视化模块

负责生成所有Plotly图表和词云。
提供丰富的交互式可视化功能来展示分析结果。
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import io
import base64
import networkx as nx
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from collections import defaultdict

# 配置日志
logger = logging.getLogger(__name__)


def create_2d_scatter(
    data: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    color_col: str = 'topic_name',
    hover_cols: Optional[List[str]] = None,
    selected_journals: Optional[List[str]] = None,
    title: str = "语义空间2D散点图"
) -> go.Figure:
    """
    创建2D语义空间散点图。
    
    Args:
        data: 包含坐标和元数据的DataFrame
        x_col, y_col: 坐标列名
        color_col: 颜色映射列名
        hover_cols: 悬停显示的列名列表
        selected_journals: 选定的期刊列表（用于绘制凸包）
        title: 图表标题
        
    Returns:
        Plotly 2D散点图对象
    """
    # 动态设置悬停列，只包含数据中实际存在的列
    if hover_cols is None:
        potential_hover_cols = ['article_title', 'journal_title', 'publication_year', 'topic_name']
        hover_cols = [col for col in potential_hover_cols if col in data.columns]
    
    # 确保必需的列存在
    required_cols = [x_col, y_col, color_col] + hover_cols
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必需的列: {missing_cols}")
    
    # 创建基础散点图
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_cols,
        title=title,
        opacity=0.8,
        height=600
    )
    
    # 添加期刊凸包和中心点
    if selected_journals and len(selected_journals) > 0:
        _add_journal_analysis_2d(fig, data, selected_journals, x_col, y_col)
    
    # 更新布局
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    logger.info(f"2D散点图创建完成，包含 {len(data)} 个点")
    
    return fig


def create_3d_scatter(
    data: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    z_col: str = 'z',
    color_col: str = 'topic_name',
    hover_cols: Optional[List[str]] = None,
    selected_journals: Optional[List[str]] = None,
    title: str = "语义空间3D散点图"
) -> go.Figure:
    """
    创建3D语义空间散点图。
    
    Args:
        data: 包含坐标和元数据的DataFrame
        x_col, y_col, z_col: 坐标列名
        color_col: 颜色映射列名
        hover_cols: 悬停显示的列名列表
        selected_journals: 选定的期刊列表（用于绘制凸包）
        title: 图表标题
        
    Returns:
        Plotly 3D散点图对象
    """
    # 动态设置悬停列，只包含数据中实际存在的列
    if hover_cols is None:
        potential_hover_cols = ['article_title', 'journal_title', 'publication_year', 'topic_name']
        hover_cols = [col for col in potential_hover_cols if col in data.columns]
    
    # 确保必需的列存在
    required_cols = [x_col, y_col, z_col, color_col] + hover_cols
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必需的列: {missing_cols}")
    
    # 创建基础散点图
    fig = px.scatter_3d(
        data,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        hover_data=hover_cols,
        title=title,
        opacity=0.8,
        height=600
    )
    
    # 添加期刊凸包和中心点
    if selected_journals and len(selected_journals) > 0:
        _add_journal_analysis(fig, data, selected_journals, x_col, y_col, z_col)
    
    # 更新布局
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    logger.info(f"3D散点图创建完成，包含 {len(data)} 个点")
    
    return fig


def _add_journal_analysis_2d(
    fig: go.Figure,
    data: pd.DataFrame,
    selected_journals: List[str],
    x_col: str,
    y_col: str
) -> None:
    """
    为2D散点图添加期刊分析（凸包和中心点）。
    
    Args:
        fig: Plotly图表对象
        data: 数据DataFrame
        selected_journals: 选定的期刊列表
        x_col, y_col: 坐标列名
    """
    colors = px.colors.qualitative.Set1
    
    # 创建期刊图例数据
    journal_legend_data = []
    
    for i, journal in enumerate(selected_journals):
        journal_data = data[data['journal_title'] == journal]
        
        if len(journal_data) < 3:  # 2D凸包需要至少3个点
            continue
        
        # 计算凸包
        coords = journal_data[[x_col, y_col]].values
        try:
            hull = ConvexHull(coords)
            
            # 绘制凸包边
            for simplex in hull.simplices:
                simplex = np.append(simplex, simplex[0])  # 闭合多边形
                fig.add_trace(go.Scatter(
                    x=coords[simplex, 0],
                    y=coords[simplex, 1],
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False,
                    opacity=0.3
                ))
            
            # 计算并绘制中心点
            center = coords.mean(axis=0)
            fig.add_trace(go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='star',
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='white')
                ),
                name=f'{journal} 中心',
                showlegend=False  # 不显示在主图例中
            ))
            
            # 添加到图例数据
            journal_legend_data.append({
                'journal': journal,
                'color': colors[i % len(colors)],
                'center': center,
                'count': len(journal_data)
            })
            
        except Exception as e:
            logger.warning(f"计算期刊 {journal} 的2D凸包失败: {e}")
    
    # 如果有期刊数据，创建单独的图例
    if journal_legend_data:
        _create_journal_legend(fig, journal_legend_data)


def _add_journal_analysis(
    fig: go.Figure,
    data: pd.DataFrame,
    selected_journals: List[str],
    x_col: str,
    y_col: str,
    z_col: str
) -> None:
    """
    为3D散点图添加期刊分析（凸包和中心点）。
    
    Args:
        fig: Plotly图表对象
        data: 数据DataFrame
        selected_journals: 选定的期刊列表
        x_col, y_col, z_col: 坐标列名
    """
    colors = px.colors.qualitative.Set1
    
    # 创建期刊图例数据
    journal_legend_data = []
    
    for i, journal in enumerate(selected_journals):
        journal_data = data[data['journal_title'] == journal]
        
        if len(journal_data) < 4:  # 凸包需要至少4个点
            continue
        
        # 计算凸包
        coords = journal_data[[x_col, y_col, z_col]].values
        try:
            hull = ConvexHull(coords)
            
            # 绘制凸包边
            for simplex in hull.simplices:
                simplex = np.append(simplex, simplex[0])  # 闭合多边形
                fig.add_trace(go.Scatter3d(
                    x=coords[simplex, 0],
                    y=coords[simplex, 1],
                    z=coords[simplex, 2],
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False,
                    opacity=0.3
                ))
            
            # 计算并绘制中心点
            center = coords.mean(axis=0)
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='star',
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='white')
                ),
                name=f'{journal} 中心',
                showlegend=False  # 不显示在主图例中
            ))
            
            # 添加到图例数据
            journal_legend_data.append({
                'journal': journal,
                'color': colors[i % len(colors)],
                'center': center,
                'count': len(journal_data)
            })
            
        except Exception as e:
            logger.warning(f"计算期刊 {journal} 的凸包失败: {e}")
    
    # 如果有期刊数据，创建单独的图例
    if journal_legend_data:
        _create_journal_legend(fig, journal_legend_data, is_3d=True)


def _create_journal_legend(
    fig: go.Figure,
    journal_legend_data: List[Dict[str, Any]],
    is_3d: bool = False
) -> None:
    """
    创建期刊图例，将其放置在图表外部。
    
    Args:
        fig: Plotly图表对象
        journal_legend_data: 期刊图例数据
        is_3d: 是否为3D图表
    """
    if not journal_legend_data:
        return
    
    # 创建图例表格数据
    legend_text = []
    legend_colors = []
    
    for data in journal_legend_data:
        # 格式化期刊信息：期刊名称 (文献数量)
        legend_text.append(f"{data['journal']} ({data['count']})")
        legend_colors.append(data['color'])
    
    # 创建子图布局
    if is_3d:
        # 3D图表使用右侧图例
        fig.update_layout(
            annotations=[
                dict(
                    text="期刊图例",
                    x=1.15,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
            ]
        )
        
        # 添加图例项
        y_positions = np.linspace(0.85, 0.05, len(legend_text))
        
        for i, (text, color) in enumerate(zip(legend_text, legend_colors)):
            fig.add_annotation(
                dict(
                    x=1.15,
                    y=y_positions[i],
                    xref="paper",
                    yref="paper",
                    text=f"■ {text}",
                    showarrow=False,
                    font=dict(size=10, color=color),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                )
            )
    else:
        # 2D图表使用底部图例
        fig.update_layout(
            annotations=[
                dict(
                    text="期刊图例",
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
            ]
        )
        
        # 添加图例项（水平排列）
        x_positions = np.linspace(0.05, 0.95, len(legend_text))
        
        for i, (text, color) in enumerate(zip(legend_text, legend_colors)):
            fig.add_annotation(
                dict(
                    x=x_positions[i],
                    y=-0.2,
                    xref="paper",
                    yref="paper",
                    text=f"■ {text}",
                    showarrow=False,
                    font=dict(size=10, color=color),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                )
            )
    
    # 调整图表边距以容纳图例
    if is_3d:
        fig.update_layout(margin=dict(r=200))
    else:
        fig.update_layout(margin=dict(b=100))


def create_topic_distribution(
    data: pd.DataFrame,
    topic_col: str = 'topic_name',
    title: str = "主题分布"
) -> go.Figure:
    """
    创建主题分布条形图。
    
    Args:
        data: 包含主题信息的DataFrame
        topic_col: 主题列名
        title: 图表标题
        
    Returns:
        Plotly条形图对象
    """
    if topic_col not in data.columns:
        raise ValueError(f"数据中缺少列: {topic_col}")
    
    # 统计主题分布
    topic_counts = data[topic_col].value_counts().sort_values(ascending=True)
    
    # 创建条形图
    fig = px.bar(
        x=topic_counts.values,
        y=topic_counts.index,
        orientation='h',
        title=title,
        labels={'x': '文献数量', 'y': '主题'}
    )
    
    # 更新布局
    fig.update_layout(
        xaxis_title="文献数量",
        yaxis_title="主题",
        height=max(400, len(topic_counts) * 30),
        margin=dict(l=150, r=20, t=40, b=40)
    )
    
    logger.info(f"主题分布图创建完成，包含 {len(topic_counts)} 个主题")
    
    return fig


def create_wordcloud(
    topic_model,
    topic_id: int,
    max_words: int = 100,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white',
    topics_df: Optional[pd.DataFrame] = None
) -> str:
    """
    为指定主题生成词云图。
    
    Args:
        topic_model: BERTopic模型
        topic_id: 主题ID
        max_words: 最大词数
        width, height: 图像尺寸
        background_color: 背景颜色
        
    Returns:
        词云图的base64编码字符串
    """
    if topic_id == -1:
        logger.warning("不为'Unclassified'主题生成词云")
        return ""
    
    try:
        # 获取主题关键词
        keywords = topic_model.get_topic(topic_id)
        if not keywords:
            return ""
        
        # 创建词频字典
        word_freq = {word: freq for word, freq in keywords[:max_words]}
        
        # 生成词云
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(word_freq)
        
        # 转换为base64
        img_buffer = io.BytesIO()
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        
        logger.info(f"主题 {topic_id} 词云生成完成")
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"生成主题 {topic_id} 词云失败: {e}")
        return ""


def create_journal_comparison(
    data: pd.DataFrame,
    journal_col: str = 'journal_title',
    topic_col: str = 'topic_name',
    top_journals: int = 10,
    title: str = "期刊-主题分布对比"
) -> go.Figure:
    """
    创建期刊-主题分布的堆叠条形图。
    
    Args:
        data: 包含期刊和主题信息的DataFrame
        journal_col: 期刊列名
        topic_col: 主题列名
        top_journals: 显示的期刊数量
        title: 图表标题
        
    Returns:
        Plotly堆叠条形图对象
    """
    required_cols = [journal_col, topic_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必需的列: {missing_cols}")
    
    # 选择前N个期刊
    top_journals_data = data[journal_col].value_counts().head(top_journals)
    filtered_data = data[data[journal_col].isin(top_journals_data.index)]
    
    # 创建交叉表
    cross_tab = pd.crosstab(
        filtered_data[journal_col],
        filtered_data[topic_col],
        normalize='index'
    )
    
    # 转换为长格式
    cross_tab_long = cross_tab.reset_index().melt(
        id_vars=journal_col,
        var_name=topic_col,
        value_name='proportion'
    )
    
    # 创建堆叠条形图
    fig = px.bar(
        cross_tab_long,
        x=journal_col,
        y='proportion',
        color=topic_col,
        title=title,
        labels={'proportion': '比例', journal_col: '期刊'},
        height=500
    )
    
    # 更新布局
    fig.update_layout(
        xaxis_title="期刊",
        yaxis_title="主题比例",
        legend_title="主题",
        barmode='stack',
        xaxis={'categoryorder': 'total descending'}
    )
    
    # 设置y轴为百分比
    fig.update_yaxes(tickformat=".0%")
    
    logger.info(f"期刊对比图创建完成，包含 {len(top_journals_data)} 个期刊")
    
    return fig


def create_temporal_trends(
    data: pd.DataFrame,
    year_col: str = 'publication_year',
    topic_col: str = 'topic_name',
    title: str = "主题时间演变趋势"
) -> go.Figure:
    """
    创建主题随时间演变的折线图。
    
    Args:
        data: 包含年份和主题信息的DataFrame
        year_col: 年份列名
        topic_col: 主题列名
        title: 图表标题
        
    Returns:
        Plotly折线图对象
    """
    required_cols = [year_col, topic_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必需的列: {missing_cols}")
    
    # 按年份和主题统计
    year_topic_counts = data.groupby([year_col, topic_col]).size().reset_index(name='count')
    
    # 计算每年的总数
    year_totals = data.groupby(year_col).size().reset_index(name='total')
    
    # 合并并计算比例
    year_topic_counts = year_topic_counts.merge(year_totals, on=year_col)
    year_topic_counts['proportion'] = year_topic_counts['count'] / year_topic_counts['total']
    
    # 创建折线图
    fig = px.line(
        year_topic_counts,
        x=year_col,
        y='proportion',
        color=topic_col,
        title=title,
        labels={'proportion': '比例', year_col: '年份'},
        markers=True,
        height=500
    )
    
    # 更新布局
    fig.update_layout(
        xaxis_title="年份",
        yaxis_title="主题比例",
        legend_title="主题",
        hovermode='x unified'
    )
    
    # 设置y轴为百分比
    fig.update_yaxes(tickformat=".0%")
    
    logger.info(f"时间趋势图创建完成，时间范围: {data[year_col].min()}-{data[year_col].max()}")
    
    return fig


def create_topic_heatmap(
    data: pd.DataFrame,
    year_col: str = 'publication_year',
    topic_col: str = 'topic_name',
    title: str = "主题-年份热力图"
) -> go.Figure:
    """
    创建主题-年份热力图。
    
    Args:
        data: 包含年份和主题信息的DataFrame
        year_col: 年份列名
        topic_col: 主题列名
        title: 图表标题
        
    Returns:
        Plotly热力图对象
    """
    required_cols = [year_col, topic_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必需的列: {missing_cols}")
    
    # 创建交叉表
    heatmap_data = pd.crosstab(
        data[year_col],
        data[topic_col],
        normalize='index'
    )
    
    # 创建热力图
    fig = px.imshow(
        heatmap_data,
        title=title,
        labels=dict(x="主题", y="年份", color="比例"),
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    
    # 更新布局
    fig.update_layout(
        xaxis_title="主题",
        yaxis_title="年份",
        height=600
    )
    
    logger.info(f"主题热力图创建完成")
    
    return fig


def create_journal_network(
    data: pd.DataFrame,
    journal_col: str = 'journal_title',
    topic_col: str = 'topic_name',
    min_cooccurrence: int = 5,
    title: str = "期刊主题共现网络"
) -> go.Figure:
    """
    创建期刊主题共现网络图。
    
    Args:
        data: 包含期刊和主题信息的DataFrame
        journal_col: 期刊列名
        topic_col: 主题列名
        min_cooccurrence: 最小共现次数
        title: 图表标题
        
    Returns:
        Plotly网络图对象
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx未安装，无法创建网络图")
        return go.Figure()
    
    # 创建期刊-主题共现矩阵
    journal_topic_matrix = pd.crosstab(data[journal_col], data[topic_col])
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点
    for journal in journal_topic_matrix.index:
        G.add_node(journal, type='journal')
    for topic in journal_topic_matrix.columns:
        G.add_node(topic, type='topic')
    
    # 添加边（期刊-主题连接）
    for journal in journal_topic_matrix.index:
        for topic in journal_topic_matrix.columns:
            weight = journal_topic_matrix.loc[journal, topic]
            if weight >= min_cooccurrence:
                G.add_edge(journal, topic, weight=weight)
    
    # 计算布局
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 准备绘图数据
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # 创建边
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # 创建节点
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append('journal' if G.nodes[node]['type'] == 'journal' else 'topic')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=10,
            color=node_color,
            colorscale=['#1f77b4', '#ff7f0e'],
            showscale=True,
            colorbar=dict(
                thickness=15,
                len=0.5,
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2)
        )
    )
    
    # 创建图形
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002 ) ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    logger.info(f"期刊网络图创建完成，包含 {len(G.nodes())} 个节点")
    
    return fig


def create_summary_stats(
    data: pd.DataFrame,
    stats_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建数据摘要统计信息。
    
    Args:
        data: 输入数据
        stats_config: 统计配置
        
    Returns:
        统计信息字典
    """
    if stats_config is None:
        stats_config = {
            'year_col': 'publication_year',
            'journal_col': 'journal_title',
            'topic_col': 'topic_name',
            'abstract_col': 'abstract_text'
        }
    
    stats = {}
    
    try:
        # 基本统计
        stats['total_records'] = len(data)
        stats['year_range'] = {
            'min': int(data[stats_config['year_col']].min()),
            'max': int(data[stats_config['year_col']].max())
        }
        stats['journal_count'] = data[stats_config['journal_col']].nunique()
        stats['topic_count'] = data[stats_config['topic_col']].nunique()
        
        # 平均摘要长度
        stats['avg_abstract_length'] = data[stats_config['abstract_col']].str.len().mean()
        
        # 年份分布
        year_dist = data[stats_config['year_col']].value_counts().sort_index()
        stats['year_distribution'] = year_dist.to_dict()
        
        # 期刊分布（前10）
        journal_dist = data[stats_config['journal_col']].value_counts().head(10)
        stats['top_journals'] = journal_dist.to_dict()
        
        # 主题分布
        topic_dist = data[stats_config['topic_col']].value_counts()
        stats['topic_distribution'] = topic_dist.to_dict()
        
        logger.info("摘要统计信息创建完成")
        
    except Exception as e:
        logger.error(f"创建摘要统计失败: {e}")
        stats = {'error': str(e)}
    
    return stats


class VisualizationError(Exception):
    """可视化异常类"""
    pass


def validate_visualization_data(data: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    验证可视化数据的有效性。
    
    Args:
        data: 输入数据
        required_cols: 必需的列名列表
        
    Returns:
        是否有效
    """
    if data.empty:
        return False
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.warning(f"数据中缺少必需的列: {missing_cols}")
        return False
    
    return True


def get_chart_theme(theme: str = 'plotly') -> str:
    """
    获取图表主题。
    
    Args:
        theme: 主题名称
        
    Returns:
        主题字符串
    """
    available_themes = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white']
    
    if theme not in available_themes:
        logger.warning(f"不支持的主题: {theme}，使用默认主题")
        theme = 'plotly'
    
    return theme


def export_chart(fig: go.Figure, format: str = 'png', width: int = 1200, height: int = 800) -> str:
    """
    导出图表为图片。
    
    Args:
        fig: Plotly图表对象
        format: 图片格式
        width, height: 图片尺寸
        
    Returns:
        图片的base64编码字符串
    """
    try:
        img_bytes = fig.to_image(format=format, width=width, height=height)
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/{format};base64,{img_base64}"
    except Exception as e:
        logger.error(f"导出图表失败: {e}")
        return ""


# ========== 新增主题模型可视化支持 ==========

def create_classical_wordcloud(
    topics_df: pd.DataFrame,
    topic_id: int,
    max_words: int = 100,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white'
) -> str:
    """
    为经典主题模型（LDA、NMF）生成词云图。
    
    Args:
        topics_df: 主题信息DataFrame
        topic_id: 主题ID
        max_words: 最大词数
        width, height: 图像尺寸
        background_color: 背景颜色
        
    Returns:
        词云图的base64编码字符串
    """
    if topic_id == -1:
        logger.warning("不为'Unclassified'主题生成词云")
        return ""
    
    try:
        # 获取主题关键词
        from src.literature_analyzer.nlp_analysis import get_classical_topic_keywords
        keywords = get_classical_topic_keywords(topics_df, topic_id, max_words)
        
        if not keywords:
            return ""
        
        # 创建词频字典
        word_freq = {word: freq for word, freq in keywords}
        
        # 生成词云
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(word_freq)
        
        # 转换为base64
        img_buffer = io.BytesIO()
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        
        logger.info(f"经典主题 {topic_id} 词云生成完成")
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"生成经典主题 {topic_id} 词云失败: {e}")
        return ""


def create_topic_model_comparison(
    comparison_df: pd.DataFrame,
    title: str = "主题模型比较"
) -> go.Figure:
    """
    创建主题模型比较图表。
    
    Args:
        comparison_df: 模型比较结果DataFrame
        title: 图表标题
        
    Returns:
        Plotly比较图表对象
    """
    if comparison_df.empty:
        logger.warning("模型比较数据为空")
        return go.Figure()
    
    try:
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('主题数量', '文档覆盖率', '多样性得分', '平均文档数/主题'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 主题数量
        fig.add_trace(
            go.Bar(x=comparison_df['model'], y=comparison_df['topic_count'],
                   name='主题数量', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 文档覆盖率
        fig.add_trace(
            go.Bar(x=comparison_df['model'], y=comparison_df['document_coverage'],
                   name='文档覆盖率', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # 多样性得分
        fig.add_trace(
            go.Bar(x=comparison_df['model'], y=comparison_df['diversity_score'],
                   name='多样性得分', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # 平均文档数/主题
        fig.add_trace(
            go.Bar(x=comparison_df['model'], y=comparison_df['avg_documents_per_topic'],
                   name='平均文档数/主题', marker_color='lightyellow'),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=600
        )
        
        # 更新轴标签
        fig.update_xaxes(title_text="模型", row=2, col=1)
        fig.update_xaxes(title_text="模型", row=2, col=2)
        fig.update_yaxes(title_text="数量", row=1, col=1)
        fig.update_yaxes(title_text="比例", row=1, col=2)
        fig.update_yaxes(title_text="得分", row=2, col=1)
        fig.update_yaxes(title_text="平均数", row=2, col=2)
        
        # 设置y轴格式
        fig.update_yaxes(tickformat=".0%", row=1, col=2)
        fig.update_yaxes(tickformat=".2f", row=2, col=1)
        
        logger.info(f"主题模型比较图创建完成，包含 {len(comparison_df)} 个模型")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建主题模型比较图失败: {e}")
        return go.Figure()


def create_topic_similarity_heatmap(
    model_results: Dict[str, Tuple[Any, pd.DataFrame]],
    title: str = "主题模型相似性热力图"
) -> go.Figure:
    """
    创建不同主题模型之间的主题相似性热力图。
    
    Args:
        model_results: 模型结果字典
        title: 图表标题
        
    Returns:
        Plotly热力图对象
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 收集所有模型的主题分布
        model_distributions = {}
        model_names = []
        
        for model_name, (model, topics_df) in model_results.items():
            if topics_df is None or len(topics_df) == 0:
                continue
                
            # 获取主题分布
            topic_dist = topics_df.groupby('topic_id')['topic_probability'].mean()
            
            if len(topic_dist) > 0:
                model_distributions[model_name] = topic_dist
                model_names.append(model_name)
        
        if len(model_distributions) < 2:
            logger.warning("需要至少2个有效模型才能创建相似性热力图")
            return go.Figure()
        
        # 计算模型间的相似性
        similarity_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 对齐主题分布并计算相似性
                    dist1 = model_distributions[model1]
                    dist2 = model_distributions[model2]
                    
                    # 使用共同的索引
                    common_indices = dist1.index.intersection(dist2.index)
                    if len(common_indices) > 0:
                        vec1 = dist1.loc[common_indices].values.reshape(1, -1)
                        vec2 = dist2.loc[common_indices].values.reshape(1, -1)
                        similarity = cosine_similarity(vec1, vec2)[0, 0]
                        similarity_matrix[i, j] = similarity
        
        # 创建热力图
        fig = px.imshow(
            similarity_matrix,
            x=model_names,
            y=model_names,
            title=title,
            labels=dict(x="模型", y="模型", color="余弦相似性"),
            color_continuous_scale="RdYlBu_r",
            aspect="auto",
            range_color=[0, 1]
        )
        
        # 更新布局
        fig.update_layout(
            height=500,
            width=600
        )
        
        # 添加相似性数值标注
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                fig.add_annotation(
                    x=model_names[j],
                    y=model_names[i],
                    text=f"{similarity_matrix[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if similarity_matrix[i, j] < 0.5 else "white")
                )
        
        logger.info(f"主题相似性热力图创建完成，包含 {len(model_names)} 个模型")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建主题相似性热力图失败: {e}")
        return go.Figure()


def create_clustering_analysis(
    topics_df: pd.DataFrame,
    reduced_embeddings: np.ndarray,
    title: str = "聚类分析图"
) -> go.Figure:
    """
    创建聚类分析图，显示聚类结果和聚类中心。
    
    Args:
        topics_df: 主题信息DataFrame
        reduced_embeddings: 降维后的嵌入向量
        title: 图表标题
        
    Returns:
        Plotly聚类分析图对象
    """
    if topics_df is None or len(topics_df) == 0:
        logger.warning("主题数据为空")
        return go.Figure()
    
    if reduced_embeddings is None or len(reduced_embeddings) == 0:
        logger.warning("降维嵌入数据为空")
        return go.Figure()
    
    try:
        # 准备数据
        plot_data = topics_df.copy()
        
        # 添加坐标数据
        if reduced_embeddings.shape[1] >= 2:
            plot_data['x'] = reduced_embeddings[:, 0]
            plot_data['y'] = reduced_embeddings[:, 1]
        
        if reduced_embeddings.shape[1] >= 3:
            plot_data['z'] = reduced_embeddings[:, 2]
        
        # 创建散点图
        if 'z' in plot_data.columns:
            # 3D散点图
            fig = px.scatter_3d(
                plot_data,
                x='x', y='y', z='z',
                color='topic_name',
                title=title,
                opacity=0.8,
                height=600
            )
            
            # 添加聚类中心（如果存在）
            if 'cluster_centers' in topics_df.attrs:
                centers = topics_df.attrs['cluster_centers']
                if centers is not None and len(centers) > 0:
                    # 对聚类中心进行降维
                    if centers.shape[1] > reduced_embeddings.shape[1]:
                        # 如果聚类中心维度高于降维维度，使用PCA降维
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=reduced_embeddings.shape[1])
                        centers_reduced = pca.fit_transform(centers)
                    else:
                        centers_reduced = centers
                    
                    # 添加聚类中心
                    fig.add_trace(go.Scatter3d(
                        x=centers_reduced[:, 0],
                        y=centers_reduced[:, 1],
                        z=centers_reduced[:, 2] if centers_reduced.shape[1] > 2 else [0] * len(centers),
                        mode='markers',
                        marker=dict(
                            size=10,
                            symbol='diamond',
                            color='red',
                            line=dict(width=2, color='white')
                        ),
                        name='聚类中心',
                        showlegend=True
                    ))
        else:
            # 2D散点图
            fig = px.scatter(
                plot_data,
                x='x', y='y',
                color='topic_name',
                title=title,
                opacity=0.8,
                height=600
            )
            
            # 添加聚类中心（如果存在）
            if 'cluster_centers' in topics_df.attrs:
                centers = topics_df.attrs['cluster_centers']
                if centers is not None and len(centers) > 0:
                    # 对聚类中心进行降维
                    if centers.shape[1] > reduced_embeddings.shape[1]:
                        # 如果聚类中心维度高于降维维度，使用PCA降维
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        centers_reduced = pca.fit_transform(centers)
                    else:
                        centers_reduced = centers
                    
                    # 添加聚类中心
                    fig.add_trace(go.Scatter(
                        x=centers_reduced[:, 0],
                        y=centers_reduced[:, 1],
                        mode='markers',
                        marker=dict(
                            size=10,
                            symbol='diamond',
                            color='red',
                            line=dict(width=2, color='white')
                        ),
                        name='聚类中心',
                        showlegend=True
                    ))
        
        # 更新布局
        if 'z' in plot_data.columns:
            fig.update_layout(
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        else:
            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Y",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        
        logger.info(f"聚类分析图创建完成，包含 {len(plot_data)} 个点")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建聚类分析图失败: {e}")
        return go.Figure()


def create_topic_evolution(
    model_results: Dict[str, Tuple[Any, pd.DataFrame]],
    title: str = "主题演化分析"
) -> go.Figure:
    """
    创建主题演化分析图，比较不同模型的主题分布。
    
    Args:
        model_results: 模型结果字典
        title: 图表标题
        
    Returns:
        Plotly主题演化图对象
    """
    try:
        # 收集所有模型的主题分布数据
        evolution_data = []
        
        for model_name, (model, topics_df) in model_results.items():
            if topics_df is None or len(topics_df) == 0:
                continue
            
            # 统计主题分布
            topic_counts = topics_df['topic_name'].value_counts()
            
            for topic_name, count in topic_counts.items():
                evolution_data.append({
                    'model': model_name,
                    'topic': topic_name,
                    'count': count,
                    'proportion': count / len(topics_df)
                })
        
        if not evolution_data:
            logger.warning("没有有效的主题分布数据")
            return go.Figure()
        
        # 创建DataFrame
        evolution_df = pd.DataFrame(evolution_data)
        
        # 创建平行坐标图
        fig = px.parallel_categories(
            evolution_df,
            dimensions=['model', 'topic'],
            color='proportion',
            title=title,
            labels={'model': '模型', 'topic': '主题', 'proportion': '比例'},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        # 更新布局
        fig.update_layout(
            height=600,
            width=800
        )
        
        logger.info(f"主题演化分析图创建完成，包含 {len(evolution_data)} 条记录")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建主题演化分析图失败: {e}")
        return go.Figure()


def create_model_performance_radar(
    comparison_df: pd.DataFrame,
    title: str = "模型性能雷达图"
) -> go.Figure:
    """
    创建模型性能雷达图。
    
    Args:
        comparison_df: 模型比较结果DataFrame
        title: 图表标题
        
    Returns:
        Plotly雷达图对象
    """
    if comparison_df.empty:
        logger.warning("模型比较数据为空")
        return go.Figure()
    
    try:
        # 选择性能指标
        metrics = ['topic_count', 'document_coverage', 'diversity_score', 'avg_documents_per_topic']
        
        # 标准化指标（0-1范围）
        normalized_data = comparison_df.copy()
        for metric in metrics:
            if metric in normalized_data.columns:
                max_val = normalized_data[metric].max()
                min_val = normalized_data[metric].min()
                if max_val > min_val:
                    normalized_data[metric] = (normalized_data[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_data[metric] = 0.5
        
        # 创建雷达图
        fig = go.Figure()
        
        for _, row in normalized_data.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=row['model']
            ))
        
        # 更新布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title=title,
            showlegend=True,
            height=600
        )
        
        logger.info(f"模型性能雷达图创建完成，包含 {len(comparison_df)} 个模型")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建模型性能雷达图失败: {e}")
        return go.Figure()


def create_topic_keyword_network(
    topics_df: pd.DataFrame,
    min_weight: float = 0.1,
    max_nodes: int = 50,
    title: str = "主题关键词网络"
) -> go.Figure:
    """
    创建主题关键词网络图，显示主题间的关系。
    
    Args:
        topics_df: 主题信息DataFrame
        min_weight: 最小权重阈值
        max_nodes: 最大节点数量
        title: 图表标题
        
    Returns:
        Plotly网络图对象
    """
    try:
        import networkx as nx
        from sklearn.feature_extraction.text import TfidfVectorizer
        from src.literature_analyzer.nlp_analysis import get_classical_topic_keywords
    except ImportError:
        logger.warning("缺少必要依赖，无法创建主题关键词网络")
        return go.Figure()
    
    if 'topic_keywords' not in topics_df.attrs:
        logger.warning("主题数据中不包含关键词信息")
        return go.Figure()
    
    try:
        # 收集所有主题的关键词
        topic_keywords = topics_df.attrs['topic_keywords']
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加主题节点
        for topic_id in topic_keywords.keys():
            if topic_id == -1:
                continue
            G.add_node(f"Topic_{topic_id}", type='topic', size=len(topic_keywords[topic_id]))
        
        # 添加关键词节点和边
        keyword_topic_map = {}
        for topic_id, keywords in topic_keywords.items():
            if topic_id == -1:
                continue
            for word, weight in keywords:
                if weight >= min_weight:
                    if word not in keyword_topic_map:
                        keyword_topic_map[word] = []
                    keyword_topic_map[word].append((topic_id, weight))
                    
                    # 添加关键词节点
                    G.add_node(word, type='keyword', weight=weight)
                    
                    # 添加主题-关键词边
                    G.add_edge(f"Topic_{topic_id}", word, weight=weight)
        
        # 添加关键词之间的共现关系
        for word, topic_pairs in keyword_topic_map.items():
            if len(topic_pairs) > 1:
                # 对共现的主题按权重排序
                topic_pairs.sort(key=lambda x: x[1], reverse=True)
                # 只保留权重最大的前几个关系
                top_pairs = topic_pairs[:3]
                for i, (topic1, weight1) in enumerate(top_pairs):
                    for (topic2, weight2) in top_pairs[i+1:]:
                        # 计算共现权重
                        co_weight = min(weight1, weight2)
                        G.add_edge(f"Topic_{topic1}", f"Topic_{topic2}", weight=co_weight*0.5, type='cooccurrence')
        
        # 限制节点数量
        if len(G.nodes()) > max_nodes:
            # 按权重排序节点
            node_weights = []
            for node in G.nodes():
                if G.nodes[node].get('type') == 'topic':
                    weight = G.nodes[node].get('size', 0)
                else:
                    weight = G.nodes[node].get('weight', 0)
                node_weights.append((node, weight))
            
            node_weights.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, _ in node_weights[:max_nodes]]
            
            # 创建子图
            G = G.subgraph(selected_nodes)
        
        # 计算布局
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 准备绘图数据
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))
        
        # 创建边
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 创建节点
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # 设置节点颜色和大小
            if G.nodes[node].get('type') == 'topic':
                node_color.append('lightblue')
                node_size.append(15)
            else:
                node_color.append('lightcoral')
                # 根据权重设置大小
                weight = G.nodes[node].get('weight', 1)
                node_size.append(5 + weight * 10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='black')
            )
        )
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        logger.info(f"主题关键词网络图创建完成，包含 {len(G.nodes())} 个节点")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建主题关键词网络图失败: {e}")
        return go.Figure()


# ========== 新增AI功能可视化支持 ==========

def create_semantic_search_results(
    results_df: pd.DataFrame,
    query: str,
    title: str = "语义搜索结果"
) -> go.Figure:
    """
    创建语义搜索结果可视化图表。
    
    Args:
        results_df: 搜索结果DataFrame
        query: 搜索查询
        title: 图表标题
        
    Returns:
        Plotly条形图对象
    """
    if results_df.empty:
        logger.warning("搜索结果为空")
        return go.Figure()
    
    try:
        # 创建条形图显示相似度
        fig = px.bar(
            results_df,
            x='similarity',
            y='article_title',
            orientation='h',
            title=f"{title}: '{query}'",
            labels={'similarity': '相似度', 'article_title': '文献标题'},
            height=max(400, len(results_df) * 40),
            color='similarity',
            color_continuous_scale='Viridis'
        )
        
        # 更新布局
        fig.update_layout(
            xaxis_title="相似度",
            yaxis_title="文献标题",
            margin=dict(l=200, r=20, t=40, b=40),
            xaxis=dict(range=[0, 1.05])  # 相似度范围0-1
        )
        
        # 添加悬停信息
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>" +
                         "期刊: %{customdata[0]}<br>" +
                         "年份: %{customdata[1]}<br>" +
                         "相似度: %{x:.3f}<extra></extra>",
            customdata=results_df[['journal_title', 'publication_year']].values
        )
        
        logger.info(f"语义搜索结果图创建完成，包含 {len(results_df)} 个结果")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建语义搜索结果图失败: {e}")
        return go.Figure()


def create_llm_qa_visualization(
    question: str,
    answer: str,
    context_sources: List[str],
    title: str = "AI问答结果"
) -> go.Figure:
    """
    创建AI问答结果可视化。
    
    Args:
        question: 用户问题
        answer: AI生成的答案
        context_sources: 上下文来源列表
        title: 图表标题
        
    Returns:
        Plotly图表对象
    """
    try:
        # 创建子图布局
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('问题', '答案'),
            specs=[[{"type": "table"}],
                   [{"type": "table"}]],
            vertical_spacing=0.1
        )
        
        # 添加问题表格
        fig.add_trace(
            go.Table(
                header=dict(values=["问题"]),
                cells=dict(values=[[question]]),
                fill_color="lightblue"
            ),
            row=1, col=1
        )
        
        # 添加答案表格
        fig.add_trace(
            go.Table(
                header=dict(values=["答案"]),
                cells=dict(values=[[answer]]),
                fill_color="lightgreen"
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False
        )
        
        # 添加上下文来源信息
        if context_sources:
            context_text = "<br>".join([f"- {source[:100]}..." for source in context_sources[:5]])
            fig.add_annotation(
                text=f"<b>上下文来源:</b><br>{context_text}",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="lightgray",
                borderwidth=1
            )
            fig.update_layout(margin=dict(b=100))
        
        logger.info("AI问答结果可视化创建完成")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建AI问答结果可视化失败: {e}")
        return go.Figure()


def create_temporal_evolution_heatmap(
    temporal_data: pd.DataFrame,
    entity_col: str,
    year_col: str = 'publication_year',
    value_col: str = 'proportion',
    title: str = "实体时间演化热力图"
) -> go.Figure:
    """
    创建实体时间演化热力图。
    
    Args:
        temporal_data: 时间分布数据
        entity_col: 实体列名
        year_col: 年份列名
        value_col: 值列名
        title: 图表标题
        
    Returns:
        Plotly热力图对象
    """
    if temporal_data.empty:
        logger.warning("时间演化数据为空")
        return go.Figure()
    
    try:
        # 创建透视表
        pivot_data = temporal_data.pivot(
            index=entity_col,
            columns=year_col,
            values=value_col
        ).fillna(0)
        
        # 创建热力图
        fig = px.imshow(
            pivot_data,
            title=title,
            labels=dict(x="年份", y=entity_col, color=value_col),
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        
        # 更新布局
        fig.update_layout(
            height=max(400, len(pivot_data) * 20),
            width=800
        )
        
        logger.info(f"时间演化热力图创建完成，包含 {len(pivot_data)} 个实体")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建时间演化热力图失败: {e}")
        return go.Figure()


def create_keyword_treemap(
    keywords: List[Tuple[str, float]],
    title: str = "关键词树状图"
) -> go.Figure:
    """
    创建关键词树状图。
    
    Args:
        keywords: 关键词及其权重的列表
        title: 图表标题
        
    Returns:
        Plotly树状图对象
    """
    if not keywords:
        logger.warning("关键词数据为空")
        return go.Figure()
    
    try:
        # 准备数据
        keyword_df = pd.DataFrame(keywords, columns=['keyword', 'weight'])
        keyword_df = keyword_df.sort_values('weight', ascending=False).head(50)  # 限制显示数量
        
        # 创建树状图
        fig = px.treemap(
            keyword_df,
            path=[px.Constant("keywords")],
            values='weight',
            names='keyword',
            title=title,
            color='weight',
            color_continuous_scale='Viridis'
        )
        
        # 更新布局
        fig.update_layout(
            height=600,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        logger.info(f"关键词树状图创建完成，包含 {len(keyword_df)} 个关键词")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建关键词树状图失败: {e}")
        return go.Figure()


def create_cooccurrence_network(
    cooccurrence_matrix: pd.DataFrame,
    min_weight: float = 0.1,
    title: str = "实体共现网络"
) -> str:
    """
    创建交互式共现网络图（使用pyvis）。
    
    Args:
        cooccurrence_matrix: 共现矩阵DataFrame
        min_weight: 最小权重阈值
        title: 图表标题
        
    Returns:
        HTML字符串
    """
    if cooccurrence_matrix.empty:
        logger.warning("共现矩阵为空")
        return ""
    
    try:
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        entities = cooccurrence_matrix.index.tolist()
        for entity in entities:
            G.add_node(entity, size=10)
        
        # 添加边
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                weight = cooccurrence_matrix.loc[entity1, entity2]
                if weight >= min_weight:
                    G.add_edge(entity1, entity2, weight=weight, title=f"共现次数: {weight:.0f}")
        
        # 使用pyvis创建交互式网络
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(G)
        
        # 设置物理布局
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -80000,
              "springConstant": 0.001,
              "damping": 0.09
            }
          }
        }
        """)
        
        # 生成HTML
        html_path = "temp_network.html"
        net.save_graph(html_path)
        
        # 读取HTML内容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 清理临时文件
        import os
        os.remove(html_path)
        
        logger.info(f"共现网络图创建完成，包含 {len(G.nodes())} 个节点")
        
        return html_content
        
    except Exception as e:
        logger.error(f"创建共现网络图失败: {e}")
        return ""


def create_topic_llm_label_comparison(
    original_labels: List[str],
    llm_labels: List[str],
    topic_ids: List[int],
    title: str = "主题标签对比"
) -> go.Figure:
    """
    创建原始主题标签与LLM生成标签的对比图。
    
    Args:
        original_labels: 原始标签列表
        llm_labels: LLM生成的标签列表
        topic_ids: 主题ID列表
        title: 图表标题
        
    Returns:
        Plotly对比图对象
    """
    if not original_labels or not llm_labels:
        logger.warning("标签数据为空")
        return go.Figure()
    
    try:
        # 准备数据
        comparison_data = []
        for i, (orig, llm, tid) in enumerate(zip(original_labels, llm_labels, topic_ids)):
            if tid == -1:
                continue  # 跳过未分类主题
            comparison_data.append({
                'topic_id': tid,
                'original_label': orig,
                'llm_label': llm,
                'label_length_diff': len(llm) - len(orig)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('标签长度对比', '标签内容'),
            specs=[[{"type": "bar"}],
                   [{"type": "table"}]],
            vertical_spacing=0.15
        )
        
        # 添加长度对比条形图
        fig.add_trace(
            go.Bar(
                x=comparison_df['topic_id'],
                y=comparison_df['label_length_diff'],
                name='长度差异',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 添加标签内容表格
        fig.add_trace(
            go.Table(
                header=dict(values=['主题ID', '原始标签', 'LLM标签']),
                cells=dict(values=[
                    comparison_df['topic_id'],
                    comparison_df['original_label'],
                    comparison_df['llm_label']
                ]),
                fill_color="lightyellow"
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        # 更新轴标签
        fig.update_xaxes(title_text="主题ID", row=1, col=1)
        fig.update_yaxes(title_text="长度差异", row=1, col=1)
        
        logger.info(f"主题标签对比图创建完成，包含 {len(comparison_df)} 个主题")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建主题标签对比图失败: {e}")
        return go.Figure()


def create_embedding_similarity_matrix(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    sample_size: int = 100,
    title: str = "嵌入向量相似性矩阵"
) -> go.Figure:
    """
    创建嵌入向量相似性矩阵热力图。
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 标签列表（可选）
        sample_size: 采样大小（用于大型数据集）
        title: 图表标题
        
    Returns:
        Plotly热力图对象
    """
    if embeddings is None or len(embeddings) == 0:
        logger.warning("嵌入向量为空")
        return go.Figure()
    
    try:
        # 采样（如果需要）
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sampled_embeddings = embeddings[indices]
            sampled_labels = [labels[i] for i in indices] if labels else None
        else:
            sampled_embeddings = embeddings
            sampled_labels = labels
        
        # 计算余弦相似度
        similarity_matrix = cosine_similarity(sampled_embeddings)
        
        # 创建热力图
        hover_text = []
        if sampled_labels:
            for i in range(len(sampled_labels)):
                row = []
                for j in range(len(sampled_labels)):
                    row.append(f"{sampled_labels[i]} vs {sampled_labels[j]}<br>相似度: {similarity_matrix[i, j]:.3f}")
                hover_text.append(row)
        else:
            hover_text = None
        
        fig = px.imshow(
            similarity_matrix,
            title=title,
            labels=dict(x="样本", y="样本", color="余弦相似度"),
            color_continuous_scale="RdYlBu_r",
            aspect="auto",
            range_color=[0, 1]
        )
        
        # 更新布局
        fig.update_layout(
            height=600,
            width=600
        )
        
        # 添加悬停信息
        if hover_text:
            for i in range(len(sampled_labels)):
                for j in range(len(sampled_labels)):
                    fig.add_annotation(
                        x=j, y=i,
                        text=f"{similarity_matrix[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color="black" if similarity_matrix[i, j] < 0.5 else "white"),
                        xref="x", yref="y"
                    )
        
        logger.info(f"嵌入相似性矩阵创建完成，形状: {similarity_matrix.shape}")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建嵌入相似性矩阵失败: {e}")
        return go.Figure()


# ========== 新增强化语义空间可视化功能 ==========

def create_multi_dimensional_semantic_space(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    methods: List[str] = ['UMAP', 't-SNE', 'PCA'],
    title: str = "多维度语义空间对比"
) -> go.Figure:
    """
    创建多维度语义空间对比图，支持多种降维方法。
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 标签列表（可选）
        colors: 颜色列表（可选）
        methods: 降维方法列表
        title: 图表标题
        
    Returns:
        Plotly子图对象
    """
    if embeddings is None or len(embeddings) == 0:
        logger.warning("嵌入向量为空")
        return go.Figure()
    
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import umap
        
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
            title=title,
            height=400 * rows,
            showlegend=False,
            hovermode='closest'
        )
        
        # 统一坐标轴标签
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(title_text="维度1", row=i, col=j)
                fig.update_yaxes(title_text="维度2", row=i, col=j)
        
        logger.info(f"多维度语义空间对比图创建完成，包含 {len(methods)} 种降维方法")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建多维度语义空间对比图失败: {e}")
        return go.Figure()


def create_semantic_space_3d_interactive(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: str = "交互式3D语义空间",
    method: str = 'UMAP'
) -> go.Figure:
    """
    创建交互式3D语义空间可视化。
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 标签列表（可选）
        colors: 颜色列表（可选）
        title: 图表标题
        method: 降维方法
        
    Returns:
        Plotly 3D散点图对象
    """
    if embeddings is None or len(embeddings) == 0:
        logger.warning("嵌入向量为空")
        return go.Figure()
    
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import umap
        
        # 采样（如果数据量太大）
        sample_size = min(3000, len(embeddings))
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sampled_embeddings = embeddings[indices]
            sampled_labels = [labels[i] for i in indices] if labels else None
            sampled_colors = [colors[i] for i in indices] if colors else None
        else:
            sampled_embeddings = embeddings
            sampled_labels = labels
            sampled_colors = colors
        
        # 应用3D降维
        if method == 'UMAP':
            reducer = umap.UMAP(n_components=3, random_state=42)
        elif method == 't-SNE':
            reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(sampled_embeddings)-1))
        elif method == 'PCA':
            reducer = PCA(n_components=3, random_state=42)
        else:
            logger.warning(f"不支持的降维方法: {method}，使用UMAP")
            reducer = umap.UMAP(n_components=3, random_state=42)
        
        reduced_3d = reducer.fit_transform(sampled_embeddings)
        
        # 创建3D散点图
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_3d[:, 0],
            y=reduced_3d[:, 1],
            z=reduced_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=sampled_colors if sampled_colors else 'lightblue',
                opacity=0.8,
                colorscale='Viridis',
                showscale=True if sampled_colors else False,
                colorbar=dict(title="类别" if sampled_colors else None)
            ),
            text=sampled_labels if sampled_labels else None,
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<extra></extra>' if sampled_labels else
                         'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            name='语义空间'
        )])
        
        # 更新布局
        fig.update_layout(
            title=f"{title} ({method})",
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
        
        logger.info(f"交互式3D语义空间创建完成，使用{method}降维，包含{len(sampled_embeddings)}个点")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建交互式3D语义空间失败: {e}")
        return go.Figure()


def create_semantic_similarity_network(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    threshold: float = 0.7,
    max_nodes: int = 100,
    title: str = "语义相似性网络"
) -> str:
    """
    创建语义相似性网络图。
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 标签列表（可选）
        threshold: 相似性阈值
        max_nodes: 最大节点数量
        title: 图表标题
        
    Returns:
        HTML字符串（pyvis网络图）
    """
    if embeddings is None or len(embeddings) == 0:
        logger.warning("嵌入向量为空")
        return ""
    
    try:
        # 采样
        sample_size = min(max_nodes, len(embeddings))
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sampled_embeddings = embeddings[indices]
            sampled_labels = [labels[i] for i in indices] if labels else [f"Doc_{i}" for i in indices]
        else:
            sampled_embeddings = embeddings
            sampled_labels = labels if labels else [f"Doc_{i}" for i in range(len(embeddings))]
        
        # 计算相似性矩阵
        similarity_matrix = cosine_similarity(sampled_embeddings)
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for i, label in enumerate(sampled_labels):
            G.add_node(i, label=label, title=f"文档: {label}")
        
        # 添加边（基于相似性阈值）
        edge_count = 0
        for i in range(len(sampled_embeddings)):
            for j in range(i + 1, len(sampled_embeddings)):
                similarity = similarity_matrix[i, j]
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
        
    except Exception as e:
        logger.error(f"创建语义相似性网络失败: {e}")
        return ""


def create_semantic_evolution_animation(
    temporal_embeddings: Dict[int, np.ndarray],
    labels: Optional[Dict[int, List[str]]] = None,
    colors: Optional[Dict[int, List[str]]] = None,
    title: str = "语义空间演化动画"
) -> go.Figure:
    """
    创建语义空间演化动画。
    
    Args:
        temporal_embeddings: 时间步到嵌入向量的映射
        labels: 时间步到标签列表的映射（可选）
        colors: 时间步到颜色列表的映射（可选）
        title: 图表标题
        
    Returns:
        Plotly动画图对象
    """
    if not temporal_embeddings:
        logger.warning("时间序列嵌入数据为空")
        return go.Figure()
    
    try:
        import umap
        
        frames = []
        time_steps = sorted(temporal_embeddings.keys())
        
        # 为每个时间步创建帧
        for step in time_steps:
            embeddings = temporal_embeddings[step]
            step_labels = labels.get(step, None) if labels else None
            step_colors = colors.get(step, None) if colors else None
            
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
                    color=step_colors if step_colors else 'lightblue',
                    opacity=0.7,
                    showscale=True if step_colors else False
                ),
                text=step_labels if step_labels else None,
                hovertemplate='%{text}<extra></extra>' if step_labels else '<extra></extra>',
                name=f'时间步 {step}'
            )
            
            frames.append(go.Frame(data=[frame_data], name=f"frame_{step}"))
        
        # 创建初始图（使用第一个时间步）
        initial_step = time_steps[0]
        initial_embeddings = temporal_embeddings[initial_step]
        initial_labels = labels.get(initial_step, None) if labels else None
        initial_colors = colors.get(initial_step, None) if colors else None
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        initial_reduced = reducer.fit_transform(initial_embeddings)
        
        fig = go.Figure(
            data=[go.Scatter(
                x=initial_reduced[:, 0],
                y=initial_reduced[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=initial_colors if initial_colors else 'lightblue',
                    opacity=0.7,
                    showscale=True if initial_colors else False
                ),
                text=initial_labels if initial_labels else None,
                hovertemplate='%{text}<extra></extra>' if initial_labels else '<extra></extra>',
                name=f'时间步 {initial_step}'
            )],
            frames=frames
        )
        
        # 添加动画控制
        fig.update_layout(
            title=title,
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
        
    except Exception as e:
        logger.error(f"创建语义空间演化动画失败: {e}")
        return go.Figure()


def create_semantic_density_heatmap(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    grid_size: int = 50,
    title: str = "语义密度热力图"
) -> go.Figure:
    """
    创建语义密度热力图。
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 标签列表（可选）
        grid_size: 网格大小
        title: 图表标题
        
    Returns:
        Plotly热力图对象
    """
    if embeddings is None or len(embeddings) == 0:
        logger.warning("嵌入向量为空")
        return go.Figure()
    
    try:
        import umap
        from scipy.stats import gaussian_kde
        
        # 降维到2D
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        
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
            text=labels if labels else None,
            hovertemplate='%{text}<extra></extra>' if labels else '<extra></extra>',
            name='数据点'
        ))
        
        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title="维度1",
            yaxis_title="维度2",
            height=600,
            showlegend=False
        )
        
        logger.info(f"语义密度热力图创建完成，网格大小: {grid_size}x{grid_size}")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建语义密度热力图失败: {e}")
        return go.Figure()


def create_semantic_clustering_analysis(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    title: str = "语义聚类分析"
) -> go.Figure:
    """
    创建语义聚类分析图。
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 标签列表（可选）
        n_clusters: 聚类数量（可选，自动确定）
        title: 图表标题
        
    Returns:
        Plotly聚类分析图对象
    """
    if embeddings is None or len(embeddings) == 0:
        logger.warning("嵌入向量为空")
        return go.Figure()
    
    try:
        import umap
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        
        # 降维到2D
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        
        # 确定聚类数量
        if n_clusters is None:
            # 使用肘部法则确定最佳聚类数
            max_clusters = min(10, len(embeddings) // 10)
            if max_clusters < 2:
                n_clusters = 2
            else:
                inertias = []
                silhouette_scores = []
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    inertias.append(kmeans.inertia_)
                    if len(set(cluster_labels)) > 1:
                        silhouette_scores.append(silhouette_score(embeddings, cluster_labels))
                    else:
                        silhouette_scores.append(0)
                
                # 选择轮廓系数最大的k值
                n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # 执行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 计算轮廓系数
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        else:
            silhouette_avg = 0
        
        # 创建散点图
        cluster_colors = px.colors.qualitative.Set3
        
        fig = go.Figure()
        
        # 绘制每个聚类的点
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_points = reduced[mask]
            cluster_text = [labels[i] for i in range(len(labels)) if mask[i]] if labels else None
            
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=cluster_colors[cluster_id % len(cluster_colors)],
                    opacity=0.7
                ),
                text=cluster_text,
                hovertemplate=f'聚类 {cluster_id}<br>' +
                             '%{text}<br>' +
                             f'点数: {len(cluster_points)}<extra></extra>' if cluster_text else
                             f'聚类 {cluster_id}<br>点数: {len(cluster_points)}<extra></extra>',
                name=f'聚类 {cluster_id}'
            ))
        
        # 添加聚类中心
        centers_reduced = reducer.transform(kmeans.cluster_centers_)
        fig.add_trace(go.Scatter(
            x=centers_reduced[:, 0],
            y=centers_reduced[:, 1],
            mode='markers',
            marker=dict(
                size=15,
                symbol='star',
                color='red',
                line=dict(width=2, color='white')
            ),
            text=[f'中心 {i}' for i in range(n_clusters)],
            hovertemplate='聚类中心 %{text}<extra></extra>',
            name='聚类中心'
        ))
        
        # 更新布局
        fig.update_layout(
            title=f"{title} (聚类数: {n_clusters}, 轮廓系数: {silhouette_avg:.3f})",
            xaxis_title="维度1",
            yaxis_title="维度2",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        logger.info(f"语义聚类分析创建完成，聚类数: {n_clusters}, 轮廓系数: {silhouette_avg:.3f}")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建语义聚类分析失败: {e}")
        return go.Figure()


def create_semantic_drift_analysis(
    embeddings_1: np.ndarray,
    embeddings_2: np.ndarray,
    labels_1: Optional[List[str]] = None,
    labels_2: Optional[List[str]] = None,
    title: str = "语义漂移分析"
) -> go.Figure:
    """
    创建语义漂移分析图，比较两个时间点的语义空间变化。
    
    Args:
        embeddings_1: 第一个时间点的嵌入向量
        embeddings_2: 第二个时间点的嵌入向量
        labels_1: 第一个时间点的标签（可选）
        labels_2: 第二个时间点的标签（可选）
        title: 图表标题
        
    Returns:
        Plotly语义漂移分析图对象
    """
    if embeddings_1 is None or embeddings_2 is None:
        logger.warning("嵌入向量为空")
        return go.Figure()
    
    try:
        import umap
        from scipy.spatial.distance import cdist
        
        # 降维到2D
        reducer = umap.UMAP(n_components=2, random_state=42)
        
        # 合并两个时间点的数据进行降维
        combined_embeddings = np.vstack([embeddings_1, embeddings_2])
        combined_reduced = reducer.fit_transform(combined_embeddings)
        
        # 分离降维结果
        reduced_1 = combined_reduced[:len(embeddings_1)]
        reduced_2 = combined_reduced[len(embeddings_1):]
        
        # 计算质心漂移
        centroid_1 = np.mean(embeddings_1, axis=0)
        centroid_2 = np.mean(embeddings_2, axis=0)
        drift_distance = np.linalg.norm(centroid_2 - centroid_1)
        
        # 计算平均距离变化
        if len(embeddings_1) == len(embeddings_2):
            pairwise_distances = cdist(embeddings_1, embeddings_2, metric='cosine')
            avg_distance = np.mean(np.diag(pairwise_distances))
        else:
            avg_distance = None
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('时间点1', '时间点2'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 时间点1
        fig.add_trace(go.Scatter(
            x=reduced_1[:, 0],
            y=reduced_1[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='lightblue',
                opacity=0.7
            ),
            text=labels_1 if labels_1 else None,
            hovertemplate='%{text}<extra></extra>' if labels_1 else '<extra></extra>',
            name='时间点1'
        ), row=1, col=1)
        
        # 时间点2
        fig.add_trace(go.Scatter(
            x=reduced_2[:, 0],
            y=reduced_2[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='lightcoral',
                opacity=0.7
            ),
            text=labels_2 if labels_2 else None,
            hovertemplate='%{text}<extra></extra>' if labels_2 else '<extra></extra>',
            name='时间点2'
        ), row=1, col=2)
        
        # 添加质心
        centroid_1_reduced = reducer.transform(centroid_1.reshape(1, -1))
        centroid_2_reduced = reducer.transform(centroid_2.reshape(1, -1))
        
        fig.add_trace(go.Scatter(
            x=[centroid_1_reduced[0, 0]],
            y=[centroid_1_reduced[0, 1]],
            mode='markers',
            marker=dict(
                size=15,
                symbol='star',
                color='blue',
                line=dict(width=2, color='white')
            ),
            name='质心1',
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[centroid_2_reduced[0, 0]],
            y=[centroid_2_reduced[0, 1]],
            mode='markers',
            marker=dict(
                size=15,
                symbol='star',
                color='red',
                line=dict(width=2, color='white')
            ),
            name='质心2',
            showlegend=False
        ), row=1, col=2)
        
        # 更新布局
        fig.update_layout(
            title=f"{title}<br>" +
                  f"质心漂移距离: {drift_distance:.3f}" +
                  (f"<br>平均距离变化: {avg_distance:.3f}" if avg_distance else ""),
            height=500,
            showlegend=False
        )
        
        # 统一坐标轴
        fig.update_xaxes(title_text="维度1", row=1, col=1)
        fig.update_xaxes(title_text="维度1", row=1, col=2)
        fig.update_yaxes(title_text="维度2", row=1, col=1)
        fig.update_yaxes(title_text="维度2", row=1, col=2)
        
        logger.info(f"语义漂移分析创建完成，质心漂移距离: {drift_distance:.3f}")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建语义漂移分析失败: {e}")
        return go.Figure()


def create_ai_enhanced_topic_summary(
    topics_df: pd.DataFrame,
    llm_summaries: Dict[int, str],
    title: str = "AI增强主题摘要"
) -> go.Figure:
    """
    创建AI增强的主题摘要可视化。
    
    Args:
        topics_df: 主题信息DataFrame
        llm_summaries: LLM生成的主题摘要字典
        title: 图表标题
        
    Returns:
        Plotly图表对象
    """
    if topics_df.empty or not llm_summaries:
        logger.warning("主题数据或AI摘要为空")
        return go.Figure()
    
    try:
        # 准备数据
        summary_data = []
        for _, row in topics_df.iterrows():
            topic_id = row['topic_id']
            if topic_id == -1:
                continue
            if topic_id in llm_summaries:
                summary_data.append({
                    'topic_id': topic_id,
                    'topic_name': row['topic_name'],
                    'document_count': len(topics_df[topics_df['topic_id'] == topic_id]),
                    'ai_summary': llm_summaries[topic_id]
                })
        
        if not summary_data:
            logger.warning("没有有效的AI摘要数据")
            return go.Figure()
        
        summary_df = pd.DataFrame(summary_data)
        
        # 创建表格
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['主题ID', '主题名称', '文献数量', 'AI摘要'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[
                    summary_df['topic_id'],
                    summary_df['topic_name'],
                    summary_df['document_count'],
                    summary_df['ai_summary']
                ],
                fill_color='lightyellow',
                align='left',
                font=dict(size=10, color='black')
            )
        )])
        
        # 更新布局
        fig.update_layout(
            title=title,
            height=max(400, len(summary_df) * 30),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        logger.info(f"AI增强主题摘要创建完成，包含 {len(summary_df)} 个主题")
        
        return fig
        
    except Exception as e:
        logger.error(f"创建AI增强主题摘要失败: {e}")
        return go.Figure()
