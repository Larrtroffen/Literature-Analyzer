"""
文献分析应用主程序

基于Streamlit的交互式文献分析应用，专门处理Web of Science导出的Excel文件。
提供完整的数据处理、NLP分析和可视化功能。
"""

import logging
import subprocess
import sys
import os
from typing import List, Optional, Dict, Any
import pandas as pd
import streamlit as st
from streamlit import cache_data, cache_resource
import numpy as np

import subprocess
import sys
import spacy
import os

# 定义要下载的spaCy模型名称
MODEL_NAME = "en_core_web_sm"

def download_spacy_model(model_name):
    """
    尝试加载spaCy模型，如果不存在则尝试下载。
    """
    try:
        # 尝试加载模型
        nlp = spacy.load(model_name)
        print(f"✅ spaCy 模型 '{model_name}' 已加载。")
        return nlp
    except OSError:
        # 如果模型不存在，则尝试下载
        print(f"⚠️ spaCy 模型 '{model_name}' 未找到。尝试下载...")
        try:
            # 构建下载命令。使用 sys.executable 确保使用当前环境的 Python 解释器。
            command = [sys.executable, "-m", "spacy", "download", model_name]
            print(f"执行命令: {' '.join(command)}")

            # 执行命令
            # check=True: 如果命令返回非零退出码（表示失败），则抛出 CalledProcessError
            # capture_output=True: 捕获标准输出和标准错误
            # text=True: 将输出解码为文本
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            print(f"🎉 模型 '{model_name}' 下载成功！")
            if result.stdout:
                print("--- 下载输出 (STDOUT) ---")
                print(result.stdout)
            if result.stderr:
                print("--- 下载错误 (STDERR) ---")
                print(result.stderr)

            # 下载成功后再次尝试加载模型
            nlp = spacy.load(model_name)
            print(f"✅ spaCy 模型 '{model_name}' 已成功下载并加载。")
            return nlp

        except subprocess.CalledProcessError as e:
            print(f"❌ 下载模型 '{model_name}' 失败。")
            print(f"命令: {e.cmd}")
            print(f"返回码: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            sys.exit(1) # 下载失败，退出脚本
        except Exception as e:
            print(f"❌ 在下载或加载模型 '{model_name}' 过程中发生意外错误: {e}")
            sys.exit(1) # 发生其他错误，退出脚本

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from literature_analyzer.data_processing import (
    load_and_process_data,
    get_data_hash,
    get_processing_stats,
    ProcessingError
)
from literature_analyzer.nlp_analysis import (
    generate_embeddings,
    perform_umap,
    perform_topic_modeling,
    load_embedding_model,
    AVAILABLE_MODELS,
    get_model_info,
    # 新增主题分析模型
    AVAILABLE_TOPIC_MODELS,
    get_topic_model_info,
    perform_lda_topic_modeling,
    perform_nmf_topic_modeling,
    perform_kmeans_topic_modeling,
    perform_hdbscan_topic_modeling,
    perform_topic_modeling_ensemble,
    compare_topic_models,
    # 新增AI功能
    perform_semantic_search,
    answer_question_with_llm,
    generate_llm_topic_label,
    extract_keywords_advanced,
    # 主题模型工厂
    perform_topic_modeling_with_factory,
    ModelLoadError,
    EmbeddingError,
    DimensionalityReductionError,
    TopicModelError
)
from literature_analyzer.visualization import (
    create_2d_scatter,
    create_3d_scatter,
    create_topic_distribution,
    create_wordcloud,
    create_journal_comparison,
    create_temporal_trends,
    create_summary_stats,
    # 新增主题模型可视化
    create_classical_wordcloud,
    create_topic_model_comparison,
    create_topic_similarity_heatmap,
    create_clustering_analysis,
    create_topic_evolution,
    create_model_performance_radar,
    create_topic_keyword_network,
    # 新增AI功能可视化
    create_semantic_search_results,
    create_llm_qa_visualization,
    create_temporal_evolution_heatmap,
    create_keyword_treemap,
    create_cooccurrence_network,
    create_topic_llm_label_comparison,
    create_embedding_similarity_matrix,
    create_ai_enhanced_topic_summary,
    # 新增强化语义空间可视化功能
    create_multi_dimensional_semantic_space,
    create_semantic_space_3d_interactive,
    create_semantic_similarity_network,
    create_semantic_evolution_animation,
    create_semantic_density_heatmap,
    create_semantic_clustering_analysis,
    create_semantic_drift_analysis,
    VisualizationError
)

# 导入新的语义空间分析模块
from literature_analyzer.semantic_space_analyzer import (
    SemanticSpaceAnalyzer,
    TemporalSemanticAnalyzer,
    InteractiveSemanticExplorer,
    compare_dimensionality_reduction_methods,
    create_semantic_drift_comparison,
)

# 辅助函数
def build_cooccurrence_matrix(items: List[str], min_cooccurrence: int = 2) -> Dict[str, Dict[str, int]]:
    """构建共现矩阵"""
    from collections import defaultdict, Counter
    import itertools
    
    # 统计每个项目的出现
    item_counts = Counter(items)
    
    # 构建共现矩阵
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    # 这里简化处理，实际应用中可能需要更复杂的共现计算逻辑
    # 例如：根据文档中的共现关系来计算
    unique_items = list(item_counts.keys())
    
    for i, item1 in enumerate(unique_items):
        for j, item2 in enumerate(unique_items):
            if i != j:
                # 简化的共现计算（实际应用中应该基于文档共现）
                cooccurrence_count = min(item_counts[item1], item_counts[item2]) // 10
                if cooccurrence_count >= min_cooccurrence:
                    cooccurrence[item1][item2] = cooccurrence_count
    
    return dict(cooccurrence)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="文献分析应用",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
def init_session_state():
    """初始化Streamlit会话状态"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'reduced_embeddings' not in st.session_state:
        st.session_state.reduced_embeddings = None
    if 'topic_model' not in st.session_state:
        st.session_state.topic_model = None
    if 'topics_df' not in st.session_state:
        st.session_state.topics_df = None
    if 'analysis_step' not in st.session_state:
        st.session_state.analysis_step = 0
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = None

# 缓存装饰器
@cache_resource
def cached_load_model(model_name: str):
    """缓存的模型加载函数"""
    return load_embedding_model(model_name)

@cache_data
def cached_process_data(files_hash: str, uploaded_files: List):
    """缓存的数据处理函数"""
    return load_and_process_data(uploaded_files)

@cache_data
def cached_generate_embeddings(texts_hash: str, texts: List[str], model_name: str):
    """缓存的嵌入生成函数"""
    return generate_embeddings(texts, model_name)

@cache_data
def cached_perform_umap(embeddings_hash: str, embeddings: np.ndarray, n_neighbors: int, min_dist: float, n_components: int):
    """缓存的UMAP降维函数"""
    return perform_umap(embeddings, n_neighbors, min_dist, n_components)

@cache_data
def cached_perform_topic_modeling(texts_hash: str, texts: List[str], embeddings: np.ndarray, min_topic_size: int, nr_topics: Optional[int]):
    """缓存的主题建模函数"""
    return perform_topic_modeling(texts, embeddings, min_topic_size, nr_topics)

def main():
    """主应用函数"""
    # 初始化会话状态
    init_session_state()
    
    # 应用标题
    st.title("📚 文献分析应用")
    st.markdown("---")
    
    # 侧边栏 - 控制面板
    with st.sidebar:
        st.header("📊 数据加载与分析")
        
        # 第一部分：数据上传与处理
        uploaded_files = st.file_uploader(
            "上传WoS Excel文件",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="请上传从Web of Science导出的Full Record Excel文件"
        )
        
        if uploaded_files and st.button("1. 加载并处理数据", type="primary"):
            process_uploaded_data(uploaded_files)
        
        # 第二部分：嵌入与降维
        if st.session_state.analysis_step >= 1:
            st.markdown("---")
            st.subheader("🔤 嵌入与降维")
            
            # 模型选择
            model_options = list(AVAILABLE_MODELS.keys())
            selected_model = st.selectbox(
                "选择嵌入模型",
                model_options,
                format_func=lambda x: f"{x} - {get_model_info(x).get('description', '')}"
            )
            
            # 显示模型信息
            model_info = get_model_info(selected_model)
            with st.expander("模型详细信息"):
                st.write(f"**类型**: {model_info.get('type', 'N/A')}")
                st.write(f"**维度**: {model_info.get('dimensions', 'N/A')}")
                st.write(f"**最大长度**: {model_info.get('max_length', 'N/A')}")
                st.write(f"**速度**: {model_info.get('speed', 'N/A')}")
                st.write(f"**质量**: {model_info.get('quality', 'N/A')}")
                st.write(f"**推荐用途**: {model_info.get('recommended_for', 'N/A')}")
                
                # 显示训练要求提示
                if model_info.get('training_required'):
                    st.warning("⚠️ 此模型需要训练数据，将使用您的文本数据进行训练")
            
            # UMAP参数
            n_neighbors = st.slider(
                "UMAP n_neighbors",
                min_value=5,
                max_value=50,
                value=15,
                help="控制局部结构与全局结构的平衡"
            )
            
            min_dist = st.slider(
                "UMAP min_dist",
                min_value=0.0,
                max_value=0.99,
                value=0.1,
                step=0.01,
                help="控制点之间的紧密程度"
            )
            
            n_components = st.selectbox(
                "目标维度",
                [2, 3],
                format_func=lambda x: f"{x}D"
            )
            
            if st.button("2. 执行嵌入与降维", type="primary"):
                perform_embedding_and_umap(selected_model, n_neighbors, min_dist, n_components)
        
        # 第三部分：主题建模
        if st.session_state.analysis_step >= 2:
            st.markdown("---")
            st.subheader("🏷️ 主题建模")
            
            # 主题模型选择
            topic_model_options = list(AVAILABLE_TOPIC_MODELS.keys())
            selected_topic_model = st.selectbox(
                "选择主题模型",
                topic_model_options,
                format_func=lambda x: f"{x} - {get_topic_model_info(x).get('description', '')}"
            )
            
            # 显示主题模型信息
            topic_model_info = get_topic_model_info(selected_topic_model)
            with st.expander("模型详细信息"):
                st.write(f"**类型**: {topic_model_info.get('type', 'N/A')}")
                st.write(f"**算法**: {topic_model_info.get('algorithm', 'N/A')}")
                st.write(f"**优点**: {topic_model_info.get('advantages', 'N/A')}")
                st.write(f"**缺点**: {topic_model_info.get('disadvantages', 'N/A')}")
                st.write(f"**适用场景**: {topic_model_info.get('suitable_for', 'N/A')}")
            
            # 根据不同模型显示不同参数
            if selected_topic_model == "BERTopic":
                min_topic_size = st.slider(
                    "最小主题大小",
                    min_value=5,
                    max_value=100,
                    value=10,
                    help="每个主题至少包含的文档数量"
                )
                
                nr_topics_option = st.selectbox(
                    "主题数量",
                    ["auto", "10", "15", "20", "25", "30"],
                    format_func=lambda x: f"{x} 个主题" if x != "auto" else "自动确定"
                )
                
                nr_topics = None if nr_topics_option == "auto" else int(nr_topics_option)
                
                if st.button("3. 执行主题分析", type="primary"):
                    perform_topic_analysis(selected_topic_model, min_topic_size=min_topic_size, nr_topics=nr_topics)
                    
            elif selected_topic_model == "LDA":
                n_topics = st.slider(
                    "主题数量",
                    min_value=2,
                    max_value=50,
                    value=10,
                    help="LDA模型的主题数量"
                )
                
                random_state = st.number_input(
                    "随机种子",
                    min_value=0,
                    max_value=1000,
                    value=42,
                    help="确保结果可重现"
                )
                
                if st.button("3. 执行主题分析", type="primary"):
                    perform_topic_analysis(selected_topic_model, n_topics=n_topics, random_state=random_state)
                    
            elif selected_topic_model == "NMF":
                n_topics = st.slider(
                    "主题数量",
                    min_value=2,
                    max_value=50,
                    value=10,
                    help="NMF模型的主题数量"
                )
                
                random_state = st.number_input(
                    "随机种子",
                    min_value=0,
                    max_value=1000,
                    value=42,
                    help="确保结果可重现"
                )
                
                if st.button("3. 执行主题分析", type="primary"):
                    perform_topic_analysis(selected_topic_model, n_topics=n_topics, random_state=random_state)
                    
            elif selected_topic_model == "KMeans":
                n_clusters = st.slider(
                    "聚类数量",
                    min_value=2,
                    max_value=50,
                    value=10,
                    help="KMeans的聚类数量"
                )
                
                random_state = st.number_input(
                    "随机种子",
                    min_value=0,
                    max_value=1000,
                    value=42,
                    help="确保结果可重现"
                )
                
                if st.button("3. 执行主题分析", type="primary"):
                    perform_topic_analysis(selected_topic_model, n_clusters=n_clusters, random_state=random_state)
                    
            elif selected_topic_model == "HDBSCAN":
                min_cluster_size = st.slider(
                    "最小聚类大小",
                    min_value=2,
                    max_value=50,
                    value=5,
                    help="HDBSCAN的最小聚类大小"
                )
                
                min_samples = st.slider(
                    "最小样本数",
                    min_value=1,
                    max_value=20,
                    value=1,
                    help="HDBSCAN的最小样本数"
                )
                
                if st.button("3. 执行主题分析", type="primary"):
                    perform_topic_analysis(selected_topic_model, min_cluster_size=min_cluster_size, min_samples=min_samples)
                    
            elif selected_topic_model == "Ensemble":
                # 集成模型参数
                min_topic_size = st.slider(
                    "最小主题大小",
                    min_value=5,
                    max_value=100,
                    value=10,
                    help="每个主题至少包含的文档数量"
                )
                
                nr_topics_option = st.selectbox(
                    "主题数量",
                    ["auto", "10", "15", "20", "25", "30"],
                    format_func=lambda x: f"{x} 个主题" if x != "auto" else "自动确定"
                )
                
                nr_topics = None if nr_topics_option == "auto" else int(nr_topics_option)
                
                # 选择要集成的模型
                ensemble_models = st.multiselect(
                    "选择要集成的模型",
                    ["BERTopic", "LDA", "NMF", "KMeans"],
                    default=["BERTopic", "LDA"]
                )
                
                if st.button("3. 执行主题分析", type="primary"):
                    perform_topic_analysis(
                        selected_topic_model, 
                        min_topic_size=min_topic_size, 
                        nr_topics=nr_topics,
                        ensemble_models=ensemble_models
                    )
        
        # 第四部分：图表筛选
        if st.session_state.analysis_step >= 3:
            st.markdown("---")
            st.header("🔍 图表筛选")
            
            # 年份筛选
            if st.session_state.processed_data is not None:
                year_range = st.session_state.processed_data['publication_year'].agg(['min', 'max'])
                selected_years = st.slider(
                    "筛选年份范围",
                    int(year_range['min']),
                    int(year_range['max']),
                    (int(year_range['min']), int(year_range['max']))
                )
            
            # 期刊筛选
            available_journals = sorted(st.session_state.processed_data['journal_title'].unique())
            selected_journals = st.multiselect(
                "筛选期刊",
                available_journals,
                default=available_journals[:10] if len(available_journals) > 10 else available_journals
            )
            
            # 主题筛选
            if st.session_state.topics_df is not None:
                available_topics = sorted(st.session_state.topics_df['topic_name'].unique())
                selected_topics = st.multiselect(
                    "筛选主题",
                    available_topics,
                    default=[t for t in available_topics if t != "Unclassified"]
                )
            
            # 关键词搜索
            search_keywords = st.text_input("关键词搜索")
        
        # 第五部分：工具
        st.markdown("---")
        st.header("🛠️ 工具")
        
        if st.button("清除所有缓存并重置"):
            clear_all_cache()
            st.rerun()
    
    # 主内容区域
    if st.session_state.analysis_step == 0:
        show_welcome_page()
    else:
        # 创建标签页
        tabs = st.tabs([
            "📋 数据概览",
            "🌐 语义空间探索", 
            "🏷️ 主题分析",
            "📰 期刊对比",
            "📈 时间趋势",
            "🤖 AI功能",
            "📊 高级分析"
        ])
        
        # 获取筛选后的数据
        filtered_data = get_filtered_data()
        
        with tabs[0]:
            show_data_overview(filtered_data)
        
        with tabs[1]:
            show_semantic_space(filtered_data)
        
        with tabs[2]:
            show_topic_analysis(filtered_data)
        
        with tabs[3]:
            show_journal_comparison(filtered_data)
        
        with tabs[4]:
            show_temporal_trends(filtered_data)
        
        with tabs[5]:
            show_ai_functions(filtered_data)
        
        with tabs[6]:
            show_advanced_analysis(filtered_data)

def process_uploaded_data(uploaded_files: List):
    """处理上传的数据文件"""
    try:
        with st.spinner("正在处理数据..."):
            # 计算数据哈希
            data_hash = get_data_hash(uploaded_files)
            
            # 检查是否已缓存
            if st.session_state.data_hash == data_hash and st.session_state.processed_data is not None:
                st.info("使用缓存的处理结果")
            else:
                # 处理数据
                processed_data = cached_process_data(data_hash, uploaded_files)
                
                # 更新会话状态
                st.session_state.processed_data = processed_data
                st.session_state.data_hash = data_hash
                st.session_state.analysis_step = 1
                
                # 计算处理统计
                original_count = sum(len(pd.read_excel(f)) for f in uploaded_files)
                st.session_state.processing_stats = get_processing_stats(
                    pd.DataFrame({'count': [original_count]}),
                    processed_data
                )
            
            st.success("数据处理完成！")
            st.rerun()
            
    except ProcessingError as e:
        st.error(f"数据处理失败: {str(e)}")
    except Exception as e:
        st.error(f"发生未知错误: {str(e)}")
        logger.error(f"数据处理错误: {e}")

def perform_embedding_and_umap(model_name: str, n_neighbors: int, min_dist: float, n_components: int):
    """执行嵌入和降维"""
    try:
        with st.spinner("正在生成嵌入向量..."):
            texts = st.session_state.processed_data['processed_text'].tolist()
            texts_hash = hash(str(texts))
            
            # 生成嵌入向量
            embeddings = cached_generate_embeddings(texts_hash, texts, model_name)
            st.session_state.embeddings = embeddings
            
        with st.spinner("正在执行UMAP降维..."):
            # 执行UMAP降维
            embeddings_hash = hash(embeddings.tobytes())
            reduced_embeddings = cached_perform_umap(
                embeddings_hash, embeddings, n_neighbors, min_dist, n_components
            )
            st.session_state.reduced_embeddings = reduced_embeddings
            
            # 添加坐标到数据中 - 先删除已存在的坐标列以避免重复
            coord_columns = ['x', 'y', 'z']
            existing_coord_columns = [col for col in coord_columns if col in st.session_state.processed_data.columns]
            if existing_coord_columns:
                st.session_state.processed_data = st.session_state.processed_data.drop(columns=existing_coord_columns)
            
            if n_components == 3:
                st.session_state.processed_data['x'] = reduced_embeddings[:, 0]
                st.session_state.processed_data['y'] = reduced_embeddings[:, 1]
                st.session_state.processed_data['z'] = reduced_embeddings[:, 2]
            else:
                st.session_state.processed_data['x'] = reduced_embeddings[:, 0]
                st.session_state.processed_data['y'] = reduced_embeddings[:, 1]
                st.session_state.processed_data['z'] = 0
            
            st.session_state.analysis_step = 2
            st.success("嵌入与降维完成！")
            st.rerun()
            
    except (ModelLoadError, EmbeddingError, DimensionalityReductionError) as e:
        st.error(f"分析失败: {str(e)}")
    except Exception as e:
        st.error(f"发生未知错误: {str(e)}")
        logger.error(f"嵌入与降维错误: {e}")

def perform_topic_analysis(model_type: str, **kwargs):
    """执行主题分析"""
    try:
        with st.spinner(f"正在执行{model_type}主题建模..."):
            texts = st.session_state.processed_data['processed_text'].tolist()
            embeddings = st.session_state.embeddings
            texts_hash = hash(str(texts))
            
            # 根据模型类型执行不同的主题建模
            if model_type == "BERTopic":
                min_topic_size = kwargs.get('min_topic_size', 10)
                nr_topics = kwargs.get('nr_topics', None)
                topic_model, topics_df = cached_perform_topic_modeling(
                    texts_hash, texts, embeddings, min_topic_size, nr_topics
                )
                
            elif model_type == "LDA":
                n_topics = kwargs.get('n_topics', 10)
                random_state = kwargs.get('random_state', 42)
                topic_model, topics_df = perform_lda_topic_modeling(
                    texts, n_topics=n_topics, random_state=random_state
                )
                
            elif model_type == "NMF":
                n_topics = kwargs.get('n_topics', 10)
                random_state = kwargs.get('random_state', 42)
                topic_model, topics_df = perform_nmf_topic_modeling(
                    texts, n_topics=n_topics, random_state=random_state
                )
                
            elif model_type == "KMeans":
                n_clusters = kwargs.get('n_clusters', 10)
                random_state = kwargs.get('random_state', 42)
                topic_model, topics_df = perform_kmeans_topic_modeling(
                    texts, embeddings, n_clusters=n_clusters, random_state=random_state
                )
                
            elif model_type == "HDBSCAN":
                min_cluster_size = kwargs.get('min_cluster_size', 5)
                min_samples = kwargs.get('min_samples', 1)
                topic_model, topics_df = perform_hdbscan_topic_modeling(
                    texts, embeddings, min_cluster_size=min_samples, min_samples=min_samples
                )
                
            elif model_type == "Ensemble":
                min_topic_size = kwargs.get('min_topic_size', 10)
                nr_topics = kwargs.get('nr_topics', None)
                ensemble_models = kwargs.get('ensemble_models', ['BERTopic', 'LDA'])
                topic_model, topics_df = perform_topic_modeling_ensemble(
                    texts, embeddings, min_topic_size=min_topic_size, nr_topics=nr_topics,
                    ensemble_models=ensemble_models
                )
                
            else:
                raise ValueError(f"不支持的主题模型类型: {model_type}")
            
            st.session_state.topic_model = topic_model
            st.session_state.topics_df = topics_df
            st.session_state.current_topic_model = model_type
            
            # 合并主题信息到主数据 - 先删除已存在的主题列以避免重复
            st.session_state.processed_data = st.session_state.processed_data.reset_index(drop=True)
            
            # 删除已存在的主题列（如果存在）
            columns_to_drop = ['topic_id', 'topic_name', 'topic_probability']
            existing_columns = [col for col in columns_to_drop if col in st.session_state.processed_data.columns]
            if existing_columns:
                st.session_state.processed_data = st.session_state.processed_data.drop(columns=existing_columns)
            
            # 添加新的主题列
            st.session_state.processed_data = pd.concat([
                st.session_state.processed_data,
                topics_df[['topic_id', 'topic_name', 'topic_probability']]
            ], axis=1)
            
            st.session_state.analysis_step = 3
            st.success(f"{model_type}主题分析完成！")
            st.rerun()
            
    except TopicModelError as e:
        st.error(f"主题建模失败: {str(e)}")
    except Exception as e:
        st.error(f"发生未知错误: {str(e)}")
        logger.error(f"主题分析错误: {e}")

def get_filtered_data() -> pd.DataFrame:
    """获取应用筛选条件后的数据"""
    if st.session_state.processed_data is None:
        return pd.DataFrame()
    
    filtered_data = st.session_state.processed_data.copy()
    
    # 年份筛选
    if 'selected_years' in st.session_state:
        year_min, year_max = st.session_state.selected_years
        filtered_data = filtered_data[
            (filtered_data['publication_year'] >= year_min) &
            (filtered_data['publication_year'] <= year_max)
        ]
    
    # 期刊筛选
    if 'selected_journals' in st.session_state and st.session_state.selected_journals:
        filtered_data = filtered_data[
            filtered_data['journal_title'].isin(st.session_state.selected_journals)
        ]
    
    # 主题筛选
    if 'selected_topics' in st.session_state and st.session_state.selected_topics:
        filtered_data = filtered_data[
            filtered_data['topic_name'].isin(st.session_state.selected_topics)
        ]
    
    # 关键词搜索
    if 'search_keywords' in st.session_state and st.session_state.search_keywords:
        keywords = st.session_state.search_keywords.lower()
        mask = (
            filtered_data['article_title'].str.lower().str.contains(keywords, na=False) |
            filtered_data['abstract_text'].str.lower().str.contains(keywords, na=False)
        )
        filtered_data = filtered_data[mask]
    
    return filtered_data

def show_welcome_page():
    """显示欢迎页面"""
    st.markdown("""
    ## 🎯 欢迎使用文献分析应用
    
    这是一个专门为科研人员设计的交互式文献分析工具，能够帮助您：
    
    - 🔍 **深度分析**：从Web of Science导出的文献数据中发现隐藏的模式和趋势
    - 📊 **可视化探索**：通过丰富的交互式图表直观理解文献结构
    - 🏷️ **主题发现**：自动识别研究主题和热点领域
    - 📰 **期刊对比**：分析不同期刊的学术特色和内容差异
    
    ### 🚀 快速开始
    
    1. **上传数据**：在左侧边栏上传一个或多个WoS Excel文件
    2. **数据处理**：点击"加载并处理数据"按钮
    3. **嵌入降维**：选择模型和参数，执行嵌入与降维
    4. **主题分析**：配置主题参数，执行主题建模
    5. **探索结果**：在不同标签页中查看分析结果
    
    ### 📋 数据要求
    
    确保您的Excel文件包含以下必需列：
    - `Article Title`（文章标题）
    - `Source Title`（期刊标题）
    - `Publication Year`（发表年份）
    - `Abstract`（摘要）
    
    ### 💡 使用提示
    
    - 支持同时上传多个Excel文件，系统会自动合并
    - 所有分析步骤都支持参数调整和实时预览
    - 提供丰富的筛选和搜索功能
    - 可以导出完整的分析结果
    
    ---
    
    *开始您的文献分析之旅吧！*
    """)

def show_data_overview(data: pd.DataFrame):
    """显示数据概览"""
    st.subheader("📋 数据概览")
    
    if data.empty:
        st.warning("没有可显示的数据")
        return
    
    # 处理统计
    if st.session_state.processing_stats:
        stats = st.session_state.processing_stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("原始记录数", stats['original_count'])
        with col2:
            st.metric("处理后记录数", stats['processed_count'])
        with col3:
            retention_rate = stats['retention_rate'] * 100
            st.metric("保留率", f"{retention_rate:.1f}%")
        with col4:
            st.metric("期刊数量", stats['journal_count'])
        
        st.markdown("---")
    
    # 数据表格
    st.subheader("📄 数据预览")
    
    # 选择显示的列
    display_columns = [
        'article_title', 'journal_title', 'publication_year', 
        'topic_name', 'topic_probability'
    ]
    available_columns = [col for col in display_columns if col in data.columns]
    
    if available_columns:
        # 显示前10行
        display_data = data[available_columns].head(10)
        st.dataframe(display_data, use_container_width=True)
        
        # 数据下载
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载完整数据 (CSV)",
            data=csv,
            file_name='literature_analysis_results.csv',
            mime='text/csv'
        )
    else:
        st.warning("没有可显示的数据列")

def show_semantic_space(data: pd.DataFrame):
    """显示语义空间探索"""
    st.subheader("🌐 语义空间探索")
    
    if data.empty or 'x' not in data.columns:
        st.warning("请先完成嵌入与降维分析")
        return
    
    # 检查是2D还是3D数据
    is_3d = 'z' in data.columns and data['z'].nunique() > 1
    
    # 颜色映射选择 - 根据可用的列动态调整选项
    color_options = []
    if 'topic_name' in data.columns:
        color_options.append('topic_name')
    if 'journal_title' in data.columns:
        color_options.append('journal_title')
    if 'publication_year' in data.columns:
        color_options.append('publication_year')
    
    # 如果没有主题信息，添加提示
    if 'topic_name' not in data.columns:
        st.info("💡 提示：完成主题分析后可以按主题进行颜色映射")
    
    if color_options:
        color_mapping = st.selectbox(
            "选择颜色映射",
            color_options,
            key="color_mapping_semantic"
        )
        
        # 获取选定的期刊（用于凸包分析）
        selected_journals = st.session_state.get('selected_journals', [])
        
        # 创建散点图
        try:
            if is_3d:
                fig = create_3d_scatter(
                    data,
                    color_col=color_mapping,
                    selected_journals=selected_journals
                )
            else:
                # 创建2D散点图
                fig = create_2d_scatter(
                    data,
                    color_col=color_mapping,
                    selected_journals=selected_journals
                )
            st.plotly_chart(fig, use_container_width=True)
            
        except VisualizationError as e:
            st.error(f"图表生成失败: {str(e)}")
    else:
        st.warning("没有可用的颜色映射选项，请确保数据中包含期刊或年份信息")

def show_topic_analysis(data: pd.DataFrame):
    """显示主题分析"""
    st.subheader("🏷️ 主题分析")
    
    if data.empty or 'topic_name' not in data.columns:
        st.warning("请先完成主题分析")
        return
    
    # 主题分布图
    st.subheader("📊 主题分布")
    try:
        topic_fig = create_topic_distribution(data)
        st.plotly_chart(topic_fig, use_container_width=True)
    except VisualizationError as e:
        st.error(f"主题分布图生成失败: {str(e)}")
    
    # 主题词云
    st.subheader("☁️ 主题词云")
    
    if st.session_state.topic_model is not None and st.session_state.topics_df is not None:
        # 获取可用主题（排除Unclassified）
        available_topics = data['topic_name'].unique()
        available_topics = [t for t in available_topics if t != "Unclassified"]
        
        if available_topics:
            selected_topic = st.selectbox("选择主题", available_topics)
            
            # 获取主题ID - 根据不同的模型类型使用不同的方法
            current_model = st.session_state.current_topic_model
            
            if current_model == "BERTopic":
                # BERTopic模型有get_topic_info方法
                topic_info = st.session_state.topic_model.get_topic_info()
                topic_row = topic_info[topic_info['Name'].str.contains(selected_topic.split()[0])]
                if not topic_row.empty:
                    topic_id = topic_row.iloc[0]['Topic']
                else:
                    topic_id = -1
            else:
                # 其他模型：从topics_df中获取主题ID
                topic_row = st.session_state.topics_df[
                    st.session_state.topics_df['topic_name'] == selected_topic
                ]
                if not topic_row.empty:
                    topic_id = topic_row.iloc[0]['topic_id']
                else:
                    topic_id = -1
            
            if topic_id != -1:
                # 生成词云 - 根据模型类型使用不同的方法
                try:
                    if current_model == "BERTopic":
                        wordcloud_b64 = create_wordcloud(st.session_state.topic_model, topic_id)
                    else:
                        # 对于经典主题模型，使用topics_df中的关键词信息
                        wordcloud_b64 = create_classical_wordcloud(
                            st.session_state.topics_df, topic_id, current_model
                        )
                    
                    if wordcloud_b64:
                        st.image(wordcloud_b64, use_container_width=True)
                    else:
                        st.warning("无法生成词云")
                except Exception as e:
                    st.warning(f"生成词云时出错: {e}")
            else:
                st.warning("无法找到选定的主题")
        else:
            st.warning("没有可用的主题")
    else:
        st.warning("主题模型未加载或主题数据不可用")

def show_journal_comparison(data: pd.DataFrame):
    """显示期刊对比"""
    st.subheader("📰 期刊对比")
    
    if data.empty or 'topic_name' not in data.columns:
        st.warning("请先完成主题分析")
        return
    
    try:
        # 期刊-主题分布图
        journal_fig = create_journal_comparison(data)
        st.plotly_chart(journal_fig, use_container_width=True)
        
    except VisualizationError as e:
        st.error(f"期刊对比图生成失败: {str(e)}")

def show_temporal_trends(data: pd.DataFrame):
    """显示时间趋势"""
    st.subheader("📈 时间趋势")
    
    if data.empty or 'topic_name' not in data.columns:
        st.warning("请先完成主题分析")
        return
    
    try:
        # 时间趋势图
        temporal_fig = create_temporal_trends(data)
        st.plotly_chart(temporal_fig, use_container_width=True)
        
    except VisualizationError as e:
        st.error(f"时间趋势图生成失败: {str(e)}")

def clear_all_cache():
    """清除所有缓存"""
    # 清除Streamlit缓存
    cache_data.clear()
    cache_resource.clear()
    
    # 清除会话状态
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # 重新初始化
    init_session_state()
    
    st.success("所有缓存已清除，应用已重置")

def show_ai_functions(data: pd.DataFrame):
    """显示AI功能页面"""
    st.subheader("🤖 AI智能分析功能")
    
    if data.empty:
        st.warning("没有可分析的数据")
        return
    
    # 创建AI功能子标签页
    ai_tabs = st.tabs([
        "🔍 语义搜索",
        "💬 AI问答",
        "🏷️ 主题标签优化"
    ])
    
    with ai_tabs[0]:
        show_semantic_search(data)
    
    with ai_tabs[1]:
        show_ai_qa(data)
    
    with ai_tabs[2]:
        show_topic_label_optimization(data)

def show_semantic_search(data: pd.DataFrame):
    """显示语义搜索功能"""
    st.subheader("🔍 语义搜索")
    
    if 'embeddings' not in st.session_state or st.session_state.embeddings is None:
        st.warning("请先完成嵌入分析")
        return
    
    # 搜索输入
    search_query = st.text_input(
        "输入搜索查询",
        placeholder="输入您想要搜索的关键词或问题..."
    )
    
    if search_query and st.button("执行语义搜索", type="primary"):
        try:
            with st.spinner("正在执行语义搜索..."):
                # 执行语义搜索
                results_df = perform_semantic_search(
                    search_query, 
                    st.session_state.embeddings,
                    data
                )
                
                if not results_df.empty:
                    # 显示搜索结果
                    st.success(f"找到 {len(results_df)} 条相关文献")
                    
                    # 创建搜索结果可视化
                    fig = create_semantic_search_results(results_df, search_query)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示详细结果
                    st.subheader("📄 搜索结果详情")
                    for idx, row in results_df.head(5).iterrows():
                        with st.expander(f"{row['article_title']} (相似度: {row['similarity']:.3f})"):
                            st.write(f"**期刊**: {row['journal_title']}")
                            st.write(f"**年份**: {row['publication_year']}")
                            st.write(f"**摘要**: {row['abstract_text'][:300]}...")
                else:
                    st.warning("未找到相关文献")
                    
        except Exception as e:
            st.error(f"语义搜索失败: {e}")

def show_ai_qa(data: pd.DataFrame):
    """显示AI问答功能"""
    st.subheader("💬 AI智能问答")
    
    # 问题输入
    question = st.text_area(
        "输入您的问题",
        placeholder="例如：这个研究领域的主要研究方向是什么？",
        height=100
    )
    
    if question and st.button("获取AI回答", type="primary"):
        try:
            with st.spinner("AI正在思考..."):
                # 准备上下文（使用前50篇文献的摘要）
                context_texts = data['abstract_text'].dropna().head(50).tolist()
                context_sources = data['article_title'].head(50).tolist()
                
                # 执行AI问答
                answer = answer_question_with_llm(question, context_texts)
                
                # 显示结果
                fig = create_llm_qa_visualization(question, answer, context_sources)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"AI问答失败: {e}")
            st.info("请确保已配置LLM API密钥")

def show_topic_label_optimization(data: pd.DataFrame):
    """显示主题标签优化功能"""
    st.subheader("🏷️ 主题标签优化")
    
    if 'topic_name' not in data.columns:
        st.warning("请先完成主题分析")
        return
    
    # 选择要优化的主题
    available_topics = data['topic_name'].unique()
    available_topics = [t for t in available_topics if t != "Unclassified"]
    
    if available_topics:
        selected_topic = st.selectbox("选择要优化的主题", available_topics)
        
        if st.button("生成优化标签", type="primary"):
            try:
                with st.spinner("AI正在生成优化标签..."):
                    # 获取主题的关键词和代表性文档
                    topic_data = data[data['topic_name'] == selected_topic]
                    topic_texts = topic_data['processed_text'].tolist()
                    
                    # 准备关键词和文档片段
                    topic_keywords = [(selected_topic, 1.0)]  # 简化处理，使用主题名称作为关键词
                    representative_docs_snippets = topic_texts[:5]  # 取前5个文档作为代表
                    
                    # 创建LLM客户端
                    try:
                        from literature_analyzer.ai_models import LLMClient
                        llm_client = LLMClient()
                    except Exception as e:
                        st.error(f"无法创建LLM客户端: {e}")
                        return
                    
                    # 生成优化标签
                    optimized_label = generate_llm_topic_label(
                        topic_keywords, 
                        representative_docs_snippets, 
                        llm_client
                    )
                    
                    # 显示对比结果
                    original_labels = [selected_topic]
                    llm_labels = [optimized_label]
                    topic_ids = [1]  # 简化处理
                    
                    fig = create_topic_llm_label_comparison(original_labels, llm_labels, topic_ids)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"标签优化失败: {e}")
                st.info("请确保已配置LLM API密钥")
    else:
        st.warning("没有可用的主题")

def show_advanced_analysis(data: pd.DataFrame):
    """显示高级分析页面"""
    st.subheader("📊 高级分析功能")
    
    if data.empty:
        st.warning("没有可分析的数据")
        return
    
    # 创建高级分析子标签页
    advanced_tabs = st.tabs([
        "🔑 关键词分析",
        "🕸️ 共现网络",
        "📊 模型比较",
        "🔄 时间演化",
        "🌐 高级语义空间"
    ])
    
    with advanced_tabs[0]:
        show_keyword_analysis(data)
    
    with advanced_tabs[1]:
        show_cooccurrence_network(data)
    
    with advanced_tabs[2]:
        show_model_comparison(data)
    
    with advanced_tabs[3]:
        show_temporal_evolution(data)
    
    with advanced_tabs[4]:
        show_advanced_semantic_space(data)

def show_keyword_analysis(data: pd.DataFrame):
    """显示关键词分析功能"""
    st.subheader("🔑 高级关键词分析")
    
    # 关键词提取方法选择
    method = st.selectbox(
        "选择关键词提取方法",
        ["YAKE", "KeyBERT", "TF-IDF"]
    )
    
    if st.button("提取关键词", type="primary"):
        try:
            with st.spinner("正在提取关键词..."):
                # 提取关键词
                keywords = extract_keywords_advanced(
                    data['processed_text'].tolist(),
                    method=method
                )
                
                if keywords:
                    # 创建关键词树状图
                    fig = create_keyword_treemap(keywords)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示关键词列表
                    st.subheader("📋 关键词列表")
                    keyword_df = pd.DataFrame(keywords, columns=['关键词', '权重'])
                    st.dataframe(keyword_df.head(20), use_container_width=True)
                else:
                    st.warning("未能提取到关键词")
                    
        except Exception as e:
            st.error(f"关键词提取失败: {e}")

def show_cooccurrence_network(data: pd.DataFrame):
    """显示共现网络功能"""
    st.subheader("🕸️ 实体共现网络")
    
    # 实体类型选择
    entity_type = st.selectbox(
        "选择实体类型",
        ["主题", "期刊", "年份"]
    )
    
    # 最小共现次数
    min_cooccurrence = st.slider(
        "最小共现次数",
        min_value=1,
        max_value=10,
        value=2
    )
    
    if st.button("生成共现网络", type="primary"):
        try:
            with st.spinner("正在构建共现网络..."):
                # 构建共现矩阵
                if entity_type == "主题":
                    cooccurrence_matrix = build_cooccurrence_matrix(
                        data['topic_name'].tolist(), min_cooccurrence
                    )
                elif entity_type == "期刊":
                    cooccurrence_matrix = build_cooccurrence_matrix(
                        data['journal_title'].tolist(), min_cooccurrence
                    )
                else:  # 年份
                    cooccurrence_matrix = build_cooccurrence_matrix(
                        data['publication_year'].astype(str).tolist(), min_cooccurrence
                    )
                
                # 生成交互式网络图
                network_html = create_cooccurrence_network(
                    cooccurrence_matrix, min_weight=min_cooccurrence
                )
                
                if network_html:
                    st.components.v1.html(network_html, height=600)
                else:
                    st.warning("无法生成共现网络")
                    
        except Exception as e:
            st.error(f"共现网络生成失败: {e}")

def show_model_comparison(data: pd.DataFrame):
    """显示模型比较功能"""
    st.subheader("📊 主题模型比较")
    
    if st.session_state.embeddings is None:
        st.warning("请先完成嵌入分析")
        return
    
    # 选择要比较的模型
    models_to_compare = st.multiselect(
        "选择要比较的主题模型",
        ["BERTopic", "LDA", "NMF", "KMeans"],
        default=["BERTopic", "LDA"]
    )
    
    if len(models_to_compare) >= 2 and st.button("执行模型比较", type="primary"):
        try:
            with st.spinner("正在比较主题模型..."):
                # 执行模型比较
                comparison_results = compare_topic_models(
                    data['processed_text'].tolist(),
                    st.session_state.embeddings,
                    models_to_compare
                )
                
                if comparison_results:
                    # 创建模型比较可视化
                    fig = create_topic_model_comparison(comparison_results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示详细比较结果
                    st.subheader("📋 模型性能详情")
                    results_df = pd.DataFrame(comparison_results)
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.warning("模型比较失败")
                    
        except Exception as e:
            st.error(f"模型比较失败: {e}")

def show_temporal_evolution(data: pd.DataFrame):
    """显示时间演化分析功能"""
    st.subheader("🔄 时间演化分析")
    
    if 'topic_name' not in data.columns:
        st.warning("请先完成主题分析")
        return
    
    # 选择分析类型
    analysis_type = st.selectbox(
        "选择分析类型",
        ["主题演化", "期刊演化", "关键词演化"]
    )
    
    if st.button("生成演化分析", type="primary"):
        try:
            with st.spinner("正在分析时间演化..."):
                if analysis_type == "主题演化":
                    # 计算主题的时间分布
                    temporal_data = data.groupby(['publication_year', 'topic_name']).size().reset_index(name='count')
                    temporal_data['proportion'] = temporal_data.groupby('publication_year')['count'].transform(lambda x: x / x.sum())
                    
                    fig = create_temporal_evolution_heatmap(
                        temporal_data, 'topic_name', 'publication_year', 'proportion'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif analysis_type == "期刊演化":
                    # 计算期刊的时间分布
                    temporal_data = data.groupby(['publication_year', 'journal_title']).size().reset_index(name='count')
                    temporal_data['proportion'] = temporal_data.groupby('publication_year')['count'].transform(lambda x: x / x.sum())
                    
                    fig = create_temporal_evolution_heatmap(
                        temporal_data, 'journal_title', 'publication_year', 'proportion'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # 关键词演化
                    st.info("关键词演化分析需要先提取关键词")
                    
        except Exception as e:
            st.error(f"时间演化分析失败: {e}")

def show_advanced_semantic_space(data: pd.DataFrame):
    """显示高级语义空间分析功能"""
    st.subheader("🌐 高级语义空间分析")
    
    if st.session_state.embeddings is None:
        st.warning("请先完成嵌入分析")
        return
    
    # 创建高级语义空间分析子标签页
    semantic_tabs = st.tabs([
        "📊 多维度降维对比",
        "🌐 交互式3D语义空间",
        "🕸️ 语义相似性网络",
        "📈 语义演化分析",
        "🔥 语义密度分析",
        "🎯 语义聚类分析",
        "📊 语义漂移分析"
    ])
    
    with semantic_tabs[0]:
        show_multi_dimensional_reduction(data)
    
    with semantic_tabs[1]:
        show_interactive_3d_semantic_space(data)
    
    with semantic_tabs[2]:
        show_semantic_similarity_network(data)
    
    with semantic_tabs[3]:
        show_semantic_evolution_analysis(data)
    
    with semantic_tabs[4]:
        show_semantic_density_analysis(data)
    
    with semantic_tabs[5]:
        show_semantic_clustering_analysis(data)
    
    with semantic_tabs[6]:
        show_semantic_drift_analysis(data)

def show_multi_dimensional_reduction(data: pd.DataFrame):
    """显示多维度降维对比功能"""
    st.subheader("📊 多维度降维方法对比")
    
    # 选择要对比的降维方法
    reduction_methods = st.multiselect(
        "选择降维方法",
        ["UMAP", "t-SNE", "PCA"],
        default=["UMAP", "t-SNE"]
    )
    
    if len(reduction_methods) >= 2 and st.button("生成降维对比", type="primary"):
        try:
            with st.spinner("正在执行降维对比..."):
                # 获取嵌入向量和标签
                embeddings = st.session_state.embeddings
                labels = data['article_title'].tolist()
                
                # 颜色映射
                color_options = []
                if 'topic_name' in data.columns:
                    color_options.append('topic_name')
                if 'journal_title' in data.columns:
                    color_options.append('journal_title')
                
                colors = None
                if color_options:
                    selected_color = st.selectbox("选择颜色映射", color_options)
                    colors = data[selected_color].tolist()
                
                # 生成对比图
                fig = create_multi_dimensional_semantic_space(
                    embeddings,
                    labels=labels,
                    colors=colors,
                    methods=reduction_methods
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"降维对比失败: {e}")

def show_interactive_3d_semantic_space(data: pd.DataFrame):
    """显示交互式3D语义空间功能"""
    st.subheader("🌐 交互式3D语义空间")
    
    # 选择降维方法
    reduction_method = st.selectbox(
        "选择降维方法",
        ["UMAP", "t-SNE", "PCA"],
        index=0
    )
    
    # 颜色映射选择
    color_options = []
    if 'topic_name' in data.columns:
        color_options.append('topic_name')
    if 'journal_title' in data.columns:
        color_options.append('journal_title')
    if 'publication_year' in data.columns:
        color_options.append('publication_year')
    
    if color_options:
        selected_color = st.selectbox("选择颜色映射", color_options)
        
        if st.button("生成交互式3D空间", type="primary"):
            try:
                with st.spinner("正在生成交互式3D语义空间..."):
                    # 获取嵌入向量和标签
                    embeddings = st.session_state.embeddings
                    labels = data['article_title'].tolist()
                    colors = data[selected_color].tolist()
                    
                    # 生成交互式3D图
                    fig = create_semantic_space_3d_interactive(
                        embeddings,
                        labels=labels,
                        colors=colors,
                        method=reduction_method
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"交互式3D空间生成失败: {e}")
    else:
        st.warning("没有可用的颜色映射选项")

def show_semantic_similarity_network(data: pd.DataFrame):
    """显示语义相似性网络功能"""
    st.subheader("🕸️ 语义相似性网络")
    
    # 相似性阈值
    similarity_threshold = st.slider(
        "相似性阈值",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="控制网络边的密度，值越高网络越稀疏"
    )
    
    # 最大节点数
    max_nodes = st.slider(
        "最大节点数",
        min_value=10,
        max_value=200,
        value=50,
        help="限制网络中显示的最大节点数"
    )
    
    if st.button("生成语义相似性网络", type="primary"):
        try:
            with st.spinner("正在生成语义相似性网络..."):
                # 创建语义空间分析器
                analyzer = SemanticSpaceAnalyzer(
                    st.session_state.embeddings,
                    data['article_title'].tolist()
                )
                
                # 生成网络图
                network_html = analyzer.create_semantic_network(
                    threshold=similarity_threshold,
                    max_nodes=max_nodes
                )
                
                if network_html:
                    st.components.v1.html(network_html, height=600)
                    
                    # 下载网络图
                    st.download_button(
                        label="📥 下载网络图 (HTML)",
                        data=network_html,
                        file_name='semantic_similarity_network.html',
                        mime='text/html'
                    )
                else:
                    st.warning("无法生成语义相似性网络")
                    
        except Exception as e:
            st.error(f"语义相似性网络生成失败: {e}")

def show_semantic_evolution_analysis(data: pd.DataFrame):
    """显示语义演化分析功能"""
    st.subheader("📈 语义演化分析")
    
    if 'publication_year' not in data.columns:
        st.warning("数据中缺少年份信息，无法进行演化分析")
        return
    
    # 选择时间步长
    time_step_options = ["每年", "每2年", "每3年", "每5年"]
    selected_time_step = st.selectbox("选择时间步长", time_step_options)
    
    # 获取年份范围
    year_min, year_max = data['publication_year'].min(), data['publication_year'].max()
    
    if st.button("生成语义演化分析", type="primary"):
        try:
            with st.spinner("正在分析语义演化..."):
                # 创建时间序列分析器
                temporal_analyzer = TemporalSemanticAnalyzer()
                
                # 根据时间步长分组数据
                step_mapping = {
                    "每年": 1,
                    "每2年": 2,
                    "每3年": 3,
                    "每5年": 5
                }
                step = step_mapping[selected_time_step]
                
                # 分组并添加时间步数据
                for year in range(year_min, year_max + 1, step):
                    year_data = data[
                        (data['publication_year'] >= year) & 
                        (data['publication_year'] < year + step)
                    ]
                    
                    if not year_data.empty:
                        # 获取对应的嵌入向量
                        year_indices = year_data.index.tolist()
                        year_embeddings = st.session_state.embeddings[year_indices]
                        year_titles = year_data['article_title'].tolist()
                        
                        temporal_analyzer.add_time_step(
                            f"{year}-{year+step-1}",
                            year_embeddings,
                            year_titles
                        )
                
                # 分析演化
                evolution_results = temporal_analyzer.analyze_temporal_evolution()
                
                # 显示演化指标
                st.subheader("📊 演化指标")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("平均漂移", f"{evolution_results['stability_metrics']['mean_drift']:.3f}")
                with col2:
                    st.metric("漂移趋势", evolution_results['stability_metrics']['drift_trend'])
                
                # 生成演化动画
                animation_fig = temporal_analyzer.create_evolution_animation()
                st.plotly_chart(animation_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"语义演化分析失败: {e}")

def show_semantic_density_analysis(data: pd.DataFrame):
    """显示语义密度分析功能"""
    st.subheader("🔥 语义密度分析")
    
    # 网格大小
    grid_size = st.slider(
        "网格大小",
        min_value=10,
        max_value=100,
        value=30,
        help="控制密度热力图的精度，值越大精度越高"
    )
    
    # 降维方法选择
    reduction_method = st.selectbox(
        "选择降维方法",
        ["UMAP", "t-SNE", "PCA"],
        index=0,
        key="reduction_method_3d"
    )
    
    if st.button("生成语义密度热力图", type="primary"):
        try:
            with st.spinner("正在生成语义密度热力图..."):
                # 创建语义空间分析器
                analyzer = SemanticSpaceAnalyzer(
                    st.session_state.embeddings,
                    data['article_title'].tolist()
                )
                
                # 生成密度热力图
                density_fig = analyzer.create_density_heatmap(grid_size=grid_size)
                st.plotly_chart(density_fig, use_container_width=True)
                
                # 显示统计信息
                stats = analyzer.get_semantic_statistics()
                st.subheader("📊 语义空间统计")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("嵌入维度", stats['embedding_dimension'])
                    st.metric("文档数量", stats['num_documents'])
                with col2:
                    st.metric("平均相似度", f"{stats['mean_similarity']:.3f}")
                    st.metric("相似度标准差", f"{stats['std_similarity']:.3f}")
                
        except Exception as e:
            st.error(f"语义密度分析失败: {e}")

def show_semantic_clustering_analysis(data: pd.DataFrame):
    """显示语义聚类分析功能"""
    st.subheader("🎯 语义聚类分析")
    
    # 聚类方法选择
    clustering_method = st.selectbox(
        "选择聚类方法",
        ["KMeans", "DBSCAN", "Agglomerative"],
        index=0,
        key="clustering_method_semantic"
    )
    
    # 聚类数量（对于需要指定数量的方法）
    if clustering_method in ["KMeans", "Agglomerative"]:
        n_clusters = st.slider(
            "聚类数量",
            min_value=2,
            max_value=20,
            value=5
        )
    
    # DBSCAN参数
    if clustering_method == "DBSCAN":
        eps = st.slider(
            "eps参数",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        min_samples = st.slider(
            "min_samples参数",
            min_value=1,
            max_value=10,
            value=2
        )
    
    if st.button("执行语义聚类分析", type="primary"):
        try:
            with st.spinner("正在执行语义聚类分析..."):
                # 创建语义空间分析器
                analyzer = SemanticSpaceAnalyzer(
                    st.session_state.embeddings,
                    data['article_title'].tolist()
                )
                
                # 执行聚类
                if clustering_method == "KMeans":
                    cluster_labels = analyzer.perform_clustering(
                        method="KMeans",
                        n_clusters=n_clusters
                    )
                elif clustering_method == "DBSCAN":
                    cluster_labels = analyzer.perform_clustering(
                        method="DBSCAN",
                        eps=eps,
                        min_samples=min_samples
                    )
                else:  # Agglomerative
                    cluster_labels = analyzer.perform_clustering(
                        method="Agglomerative",
                        n_clusters=n_clusters
                    )
                
                # 生成聚类分析图
                fig = create_semantic_clustering_analysis(
                    st.session_state.embeddings,
                    labels=data['article_title'].tolist(),
                    n_clusters=len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示聚类统计
                unique_labels = set(cluster_labels)
                n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                st.metric("发现的聚类数", n_clusters_found)
                
                # 如果有足够的聚类，显示轮廓系数
                if n_clusters_found > 1:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(st.session_state.embeddings, cluster_labels)
                    st.metric("轮廓系数", f"{silhouette:.3f}")
                
        except Exception as e:
            st.error(f"语义聚类分析失败: {e}")

def show_semantic_drift_analysis(data: pd.DataFrame):
    """显示语义漂移分析功能"""
    st.subheader("📊 语义漂移分析")
    
    if 'publication_year' not in data.columns:
        st.warning("数据中缺少年份信息，无法进行漂移分析")
        return
    
    # 选择对比的时间段
    years = sorted(data['publication_year'].unique())
    if len(years) < 2:
        st.warning("数据中年份不足，无法进行漂移分析")
        return
    
    # 时间段选择
    col1, col2 = st.columns(2)
    with col1:
        year1 = st.selectbox("选择第一个年份", years[:-1])
    with col2:
        year2 = st.selectbox("选择第二个年份", years[1:])
    
    # 漂移分析方法
    drift_method = st.selectbox(
        "选择漂移分析方法",
        ["质心漂移", "分布漂移", "成对距离"],
        index=0,
        key="drift_method_semantic"
    )
    
    if st.button("执行语义漂移分析", type="primary"):
        try:
            with st.spinner("正在分析语义漂移..."):
                # 获取两个时间段的嵌入向量
                embeddings_1 = st.session_state.embeddings[data['publication_year'] == year1]
                embeddings_2 = st.session_state.embeddings[data['publication_year'] == year2]
                
                # 创建分析器并分析漂移
                analyzer = SemanticSpaceAnalyzer(embeddings_1)
                
                method_mapping = {
                    "质心漂移": "centroid",
                    "分布漂移": "distribution",
                    "成对距离": "pairwise"
                }
                
                drift_metrics = analyzer.analyze_semantic_drift(
                    embeddings_2,
                    method=method_mapping[drift_method]
                )
                
                # 显示漂移指标
                st.subheader("📊 漂移分析结果")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("漂移距离", f"{drift_metrics['drift_distance']:.3f}")
                with col2:
                    st.metric("相对漂移", f"{drift_metrics['relative_drift']:.3f}")
                with col3:
                    st.metric("漂移方向", drift_metrics['drift_direction'])
                
                # 生成漂移对比图
                embeddings_dict = {
                    f"{year1}年": embeddings_1,
                    f"{year2}年": embeddings_2
                }
                
                drift_fig = create_semantic_drift_comparison(
                    embeddings_dict, 
                    reference_key=f"{year1}年"
                )
                st.plotly_chart(drift_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"语义漂移分析失败: {e}")

if __name__ == "__main__":
    nlp = download_spacy_model(MODEL_NAME)
    main()
