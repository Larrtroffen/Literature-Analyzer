"""
Literature Analyzer Package

一个基于Streamlit的交互式文献分析应用，专门处理从Web of Science (WoS)
导出的Full Record Excel文件，提供强大的文本分析和可视化功能。
"""

__version__ = "2.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# 数据处理模块
from .data_processing import load_and_process_data, preprocess_text

# NLP分析模块
from .nlp_analysis import (
    generate_embeddings,
    perform_umap,
    perform_topic_modeling,
    load_embedding_model,
    # 新增主题分析模型
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
    get_temporal_distribution,
    extract_keywords_advanced,
    build_cooccurrence_matrix,
    perform_topic_modeling_with_factory
)

# 主题模型工厂
from .topic_model_factory import TopicModelFactory, AbstractTopicModel

# AI模型模块
from .ai_models import LLMClient, OpenAIProvider, LocalLLMProvider

# 可视化模块
from .visualization import (
    create_3d_scatter,
    create_topic_distribution,
    create_wordcloud,
    create_journal_comparison,
    create_temporal_trends,
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
)

# 语义空间分析模块
from .semantic_space_analyzer import (
    SemanticSpaceAnalyzer,
    TemporalSemanticAnalyzer,
    InteractiveSemanticExplorer,
    compare_dimensionality_reduction_methods,
    create_semantic_drift_comparison,
)

__all__ = [
    # 数据处理模块
    "load_and_process_data",
    "preprocess_text",
    
    # NLP分析模块
    "generate_embeddings",
    "perform_umap",
    "perform_topic_modeling",
    "load_embedding_model",
    # 新增主题分析模型
    "perform_lda_topic_modeling",
    "perform_nmf_topic_modeling",
    "perform_kmeans_topic_modeling",
    "perform_hdbscan_topic_modeling",
    "perform_topic_modeling_ensemble",
    "compare_topic_models",
    # 新增AI功能
    "perform_semantic_search",
    "answer_question_with_llm",
    "generate_llm_topic_label",
    "get_temporal_distribution",
    "extract_keywords_advanced",
    "build_cooccurrence_matrix",
    "perform_topic_modeling_with_factory",
    
    # 主题模型工厂
    "TopicModelFactory",
    "AbstractTopicModel",
    
    # AI模型模块
    "LLMClient",
    "OpenAIProvider", 
    "LocalLLMProvider",
    
    # 可视化模块
    "create_3d_scatter",
    "create_topic_distribution",
    "create_wordcloud",
    "create_journal_comparison",
    "create_temporal_trends",
    # 新增主题模型可视化
    "create_classical_wordcloud",
    "create_topic_model_comparison",
    "create_topic_similarity_heatmap",
    "create_clustering_analysis",
    "create_topic_evolution",
    "create_model_performance_radar",
    "create_topic_keyword_network",
    # 新增AI功能可视化
    "create_semantic_search_results",
    "create_llm_qa_visualization",
    "create_temporal_evolution_heatmap",
    "create_keyword_treemap",
    "create_cooccurrence_network",
    "create_topic_llm_label_comparison",
    "create_embedding_similarity_matrix",
    "create_ai_enhanced_topic_summary",
    # 新增强化语义空间可视化功能
    "create_multi_dimensional_semantic_space",
    "create_semantic_space_3d_interactive",
    "create_semantic_similarity_network",
    "create_semantic_evolution_animation",
    "create_semantic_density_heatmap",
    "create_semantic_clustering_analysis",
    "create_semantic_drift_analysis",
    
    # 语义空间分析模块
    "SemanticSpaceAnalyzer",
    "TemporalSemanticAnalyzer",
    "InteractiveSemanticExplorer",
    "compare_dimensionality_reduction_methods",
    "create_semantic_drift_comparison",
]
