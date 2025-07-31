"""
NLP分析模块

负责语义嵌入、降维和主题建模功能。
提供文本的深度分析和主题发现能力。
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from gensim.models import Doc2Vec, Word2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
import torch
import hdbscan
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# 导入新的模块
from .topic_model_factory import TopicModelFactory, AbstractTopicModel
from .ai_models import LLMClient

# 配置日志
logger = logging.getLogger(__name__)

# 可用的嵌入模型
AVAILABLE_MODELS = {
    # Sentence Transformers 模型
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "text-embedding-ada-002": "text-embedding-ada-002",
    
    # Gensim 模型
    "doc2vec": "gensim_doc2vec",
    "word2vec": "gensim_word2vec", 
    "fasttext": "gensim_fasttext",
    
    # Hugging Face Transformers 模型
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "distilbert-base-uncased": "distilbert-base-uncased"
}


def load_embedding_model(model_name: str, texts: Optional[List[str]] = None) -> Any:
    """
    加载预训练的嵌入模型。
    
    支持多种类型的嵌入模型：Sentence Transformers、Gensim模型、Hugging Face Transformers。
    
    Args:
        model_name: 模型名称
        texts: 用于训练Gensim模型的文本列表（可选）
        
    Returns:
        加载的嵌入模型
        
    Raises:
        ValueError: 当模型名称无效时
        ModelLoadError: 当模型加载失败时
    """
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(f"无效的模型名称: {model_name}. 可用模型: {available}")
    
    model_path = AVAILABLE_MODELS[model_name]
    
    try:
        logger.info(f"正在加载嵌入模型: {model_name}")
        
        # Sentence Transformers 模型
        if model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "text-embedding-ada-002"]:
            model = SentenceTransformer(model_path)
        
        # Gensim 模型
        elif model_name == "doc2vec":
            if texts is None or len(texts) == 0:
                raise ValueError("Doc2Vec模型需要训练文本")
            model = _train_doc2vec_model(texts)
        
        elif model_name == "word2vec":
            if texts is None or len(texts) == 0:
                raise ValueError("Word2Vec模型需要训练文本")
            model = _train_word2vec_model(texts)
        
        elif model_name == "fasttext":
            if texts is None or len(texts) == 0:
                raise ValueError("FastText模型需要训练文本")
            model = _train_fasttext_model(texts)
        
        # Hugging Face Transformers 模型
        elif model_name in ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]:
            model = _load_transformer_model(model_name)
        
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
        
        logger.info(f"模型 {model_name} 加载成功")
        return model
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise ModelLoadError(f"无法加载模型 {model_name}: {e}") from e


def _train_doc2vec_model(texts: List[str], vector_size: int = 100, window: int = 5, 
                        min_count: int = 2, epochs: int = 100) -> Doc2Vec:
    """
    训练Doc2Vec模型。
    
    Args:
        texts: 训练文本列表
        vector_size: 向量维度
        window: 上下文窗口大小
        min_count: 最小词频
        epochs: 训练轮数
        
    Returns:
        训练好的Doc2Vec模型
    """
    logger.info("开始训练Doc2Vec模型")
    
    # 准备训练数据
    tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]
    
    # 训练模型
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=4,
        dm=1,  # 分布式内存模型
        alpha=0.025,
        min_alpha=0.00025
    )
    
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    logger.info(f"Doc2Vec模型训练完成，向量维度: {vector_size}")
    return model


def _train_word2vec_model(texts: List[str], vector_size: int = 100, window: int = 5,
                         min_count: int = 2, epochs: int = 100) -> Word2Vec:
    """
    训练Word2Vec模型。
    
    Args:
        texts: 训练文本列表
        vector_size: 向量维度
        window: 上下文窗口大小
        min_count: 最小词频
        epochs: 训练轮数
        
    Returns:
        训练好的Word2Vec模型
    """
    logger.info("开始训练Word2Vec模型")
    
    # 准备训练数据
    sentences = [text.split() for text in texts]
    
    # 训练模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=4,
        sg=1  # Skip-gram模型
    )
    
    logger.info(f"Word2Vec模型训练完成，向量维度: {vector_size}")
    return model


def _train_fasttext_model(texts: List[str], vector_size: int = 100, window: int = 5,
                         min_count: int = 2, epochs: int = 100) -> FastText:
    """
    训练FastText模型。
    
    Args:
        texts: 训练文本列表
        vector_size: 向量维度
        window: 上下文窗口大小
        min_count: 最小词频
        epochs: 训练轮数
        
    Returns:
        训练好的FastText模型
    """
    logger.info("开始训练FastText模型")
    
    # 准备训练数据
    sentences = [text.split() for text in texts]
    
    # 训练模型
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=4,
        sg=1  # Skip-gram模型
    )
    
    logger.info(f"FastText模型训练完成，向量维度: {vector_size}")
    return model


def _load_transformer_model(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    """
    加载Hugging Face Transformer模型。
    
    Args:
        model_name: 模型名称
        
    Returns:
        (tokenizer, model) 元组
    """
    logger.info(f"加载Transformer模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 设置为评估模式
    model.eval()
    
    return tokenizer, model


def generate_embeddings(
    texts: List[str], 
    model_name: str, 
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    使用指定的模型生成文本嵌入向量。
    
    支持多种嵌入模型类型，包括Sentence Transformers、Gensim和Hugging Face Transformers。
    
    Args:
        texts: 文本列表
        model_name: 嵌入模型名称
        batch_size: 批处理大小
        show_progress: 是否显示进度条
        
    Returns:
        嵌入向量矩阵 (n_texts, embedding_dim)
        
    Raises:
        ValueError: 当输入文本为空时
        EmbeddingError: 当嵌入生成失败时
    """
    if not texts:
        raise ValueError("输入文本列表不能为空")
    
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("所有文本必须是字符串")
    
    try:
        logger.info(f"开始生成嵌入向量，共 {len(texts)} 个文本，使用模型: {model_name}")
        
        # Sentence Transformers 模型
        if model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "text-embedding-ada-002"]:
            model = load_embedding_model(model_name)
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
        
        # Gensim 模型
        elif model_name in ["doc2vec", "word2vec", "fasttext"]:
            model = load_embedding_model(model_name, texts)
            embeddings = _generate_gensim_embeddings(texts, model, model_name)
        
        # Hugging Face Transformers 模型
        elif model_name in ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]:
            tokenizer, model = load_embedding_model(model_name)
            embeddings = _generate_transformer_embeddings(texts, tokenizer, model, batch_size, show_progress)
        
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
        
        logger.info(f"嵌入向量生成完成，形状: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logger.error(f"嵌入向量生成失败: {e}")
        raise EmbeddingError(f"嵌入向量生成失败: {e}") from e


def _generate_gensim_embeddings(texts: List[str], model: Any, model_name: str) -> np.ndarray:
    """
    使用Gensim模型生成嵌入向量。
    
    Args:
        texts: 文本列表
        model: Gensim模型
        model_name: 模型名称
        
    Returns:
        嵌入向量矩阵
    """
    logger.info(f"使用{model_name}生成嵌入向量")
    
    if model_name == "doc2vec":
        # Doc2Vec可以直接推断文档向量
        embeddings = []
        for text in texts:
            vector = model.infer_vector(text.split())
            embeddings.append(vector)
        return np.array(embeddings)
    
    elif model_name in ["word2vec", "fasttext"]:
        # Word2Vec和FastText需要聚合词向量
        embeddings = []
        for text in texts:
            words = text.split()
            word_vectors = []
            
            for word in words:
                if word in model.wv:
                    word_vectors.append(model.wv[word])
            
            if word_vectors:
                # 使用词向量的平均值作为文档向量
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # 如果没有找到任何词，使用零向量
                doc_vector = np.zeros(model.vector_size)
            
            embeddings.append(doc_vector)
        
        return np.array(embeddings)
    
    else:
        raise ValueError(f"不支持的Gensim模型: {model_name}")


def _generate_transformer_embeddings(
    texts: List[str], 
    tokenizer: AutoTokenizer, 
    model: AutoModel,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    使用Hugging Face Transformer模型生成嵌入向量。
    
    Args:
        texts: 文本列表
        tokenizer: 分词器
        model: Transformer模型
        batch_size: 批处理大小
        show_progress: 是否显示进度条
        
    Returns:
        嵌入向量矩阵
    """
    logger.info("使用Transformer模型生成嵌入向量")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_embeddings = []
    
    # 分批处理
    for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入向量", disable=not show_progress):
        batch_texts = texts[i:i + batch_size]
        
        # 分词
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 移动到设备
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # 生成嵌入
        with torch.no_grad():
            outputs = model(**encoded)
            
            # 使用最后一层隐藏状态的平均值作为句子嵌入
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded['attention_mask']
            
            # 计算注意力掩码的扩展版本
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # 计算掩码后的隐藏状态总和
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            
            # 计算实际token数量
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # 计算平均值
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(batch_embeddings)
    
    # 合并所有批次的嵌入
    embeddings = np.vstack(all_embeddings)
    
    return embeddings


def perform_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 3,
    random_state: int = 42
) -> np.ndarray:
    """
    使用UMAP算法对嵌入向量进行降维。
    
    Args:
        embeddings: 高维嵌入向量
        n_neighbors: UMAP n_neighbors参数
        min_dist: UMAP min_dist参数
        n_components: 目标维度 (2或3)
        random_state: 随机种子
        
    Returns:
        降维后的坐标矩阵
        
    Raises:
        ValueError: 当参数无效时
        DimensionalityReductionError: 当降维失败时
    """
    if n_components not in [2, 3]:
        raise ValueError("n_components必须是2或3")
    
    if n_neighbors < 2 or n_neighbors > len(embeddings):
        raise ValueError(f"n_neighbors必须在2和{len(embeddings)}之间")
    
    if not (0.0 <= min_dist <= 1.0):
        raise ValueError("min_dist必须在0.0和1.0之间")
    
    try:
        logger.info(f"开始UMAP降维: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}")
        
        # 创建UMAP模型
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
            metric='cosine'
        )
        
        # 执行降维
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        logger.info(f"UMAP降维完成，输出形状: {reduced_embeddings.shape}")
        
        return reduced_embeddings
        
    except Exception as e:
        logger.error(f"UMAP降维失败: {e}")
        raise DimensionalityReductionError(f"UMAP降维失败: {e}") from e


def perform_topic_modeling(
    texts: List[str],
    embeddings: np.ndarray,
    min_topic_size: int = 10,
    nr_topics: Optional[int] = None,
    verbose: bool = True
) -> Tuple[BERTopic, pd.DataFrame]:
    """
    使用BERTopic进行主题建模。
    
    Args:
        texts: 预处理后的文本列表
        embeddings: 文本嵌入向量
        min_topic_size: 最小主题大小
        nr_topics: 主题数量，None表示自动确定
        verbose: 是否显示详细信息
        
    Returns:
        (trained_model, topics_df): 训练好的BERTopic模型和主题信息DataFrame
        
    Raises:
        ValueError: 当输入参数无效时
        TopicModelError: 当主题建模失败时
    """
    if not texts or len(texts) != len(embeddings):
        raise ValueError("文本列表和嵌入向量长度必须相同且非空")
    
    if min_topic_size < 2:
        raise ValueError("min_topic_size必须大于等于2")
    
    if nr_topics is not None and nr_topics < 2:
        raise ValueError("nr_topics必须大于等于2或为None")
    
    try:
        logger.info(f"开始BERTopic主题建模: min_topic_size={min_topic_size}, nr_topics={nr_topics}")
        
        # 配置向量化器
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # 创建BERTopic模型
        topic_model = BERTopic(
            vectorizer_model=vectorizer,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            verbose=verbose,
            calculate_probabilities=True
        )
        
        # 训练模型
        topics, probabilities = topic_model.fit_transform(texts, embeddings)
        
        logger.info(f"主题建模完成，发现 {len(set(topics)) - (1 if -1 in topics else 0)} 个主题")
        
        # 创建主题信息DataFrame
        topics_df = _create_topics_dataframe(topic_model, topics, probabilities)
        
        return topic_model, topics_df
        
    except Exception as e:
        error_msg = str(e)
        if "ambiguous" in error_msg and "array" in error_msg:
            logger.error(f"主题建模失败: NumPy数组比较错误 - {error_msg}")
            raise TopicModelError(f"主题建模失败: NumPy数组比较错误，请检查输入数据格式。详细信息: {error_msg}") from e
        elif "ValueError" in error_msg or "shape" in error_msg:
            logger.error(f"主题建模失败: 数据形状不匹配 - {error_msg}")
            raise TopicModelError(f"主题建模失败: 数据形状不匹配，请确保文本和嵌入向量长度一致。详细信息: {error_msg}") from e
        else:
            logger.error(f"主题建模失败: {error_msg}")
            raise TopicModelError(f"主题建模失败: {error_msg}") from e


def _create_topics_dataframe(
    topic_model: BERTopic,
    topics: List[int],
    probabilities: np.ndarray
) -> pd.DataFrame:
    """
    创建包含主题信息的DataFrame。
    
    Args:
        topic_model: 训练好的BERTopic模型
        topics: 主题分配列表
        probabilities: 主题概率矩阵
        
    Returns:
        包含主题信息的DataFrame
    """
    # 获取主题信息
    topic_info = topic_model.get_topic_info()
    
    # 创建主题名称映射
    topic_names = {}
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            topic_names[topic_id] = "Unclassified"
        else:
            # 获取主题的前几个关键词作为名称
            words = row['Name'].split("_")[:3]  # 取前3个词
            topic_names[topic_id] = " ".join(words)
    
    # 创建DataFrame
    topics_data = []
    for i, (topic_id, prob) in enumerate(zip(topics, probabilities)):
        # 安全地获取主题概率
        if topic_id != -1 and topic_id < prob.shape[0]:
            topic_probability = float(prob[topic_id])
        else:
            topic_probability = 0.0
        
        topic_data = {
            'document_id': i,
            'topic_id': topic_id,
            'topic_name': topic_names[topic_id],
            'topic_probability': topic_probability
        }
        topics_data.append(topic_data)
    
    topics_df = pd.DataFrame(topics_data)
    
    logger.info(f"主题信息DataFrame创建完成，包含 {len(topics_df)} 条记录")
    
    return topics_df


def get_topic_keywords(topic_model: BERTopic, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    获取指定主题的关键词及其权重。
    
    Args:
        topic_model: 训练好的BERTopic模型
        topic_id: 主题ID
        top_n: 返回的关键词数量
        
    Returns:
        关键词及其权重的列表
    """
    if topic_id == -1:
        return []
    
    try:
        keywords = topic_model.get_topic(topic_id)
        return keywords[:top_n]
    except Exception as e:
        logger.warning(f"获取主题 {topic_id} 的关键词失败: {e}")
        return []


def get_topic_documents(
    topic_model: BERTopic,
    texts: List[str],
    topic_id: int,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    获取指定主题的代表性文档。
    
    Args:
        topic_model: 训练好的BERTopic模型
        texts: 原始文本列表
        topic_id: 主题ID
        top_n: 返回的文档数量
        
    Returns:
        文本及其概率的列表
    """
    if topic_id == -1:
        return []
    
    try:
        # 获取主题的文档概率
        topics, _ = topic_model.transform(texts)
        doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
        
        if not doc_indices:
            return []
        
        # 获取这些文档的概率
        probabilities = topic_model.probabilities_[doc_indices, topic_id]
        
        # 按概率排序并返回前N个
        doc_prob_pairs = list(zip(
            [texts[i] for i in doc_indices],
            probabilities
        ))
        doc_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_prob_pairs[:top_n]
        
    except Exception as e:
        logger.warning(f"获取主题 {topic_id} 的文档失败: {e}")
        return []


def calculate_topic_diversity(topic_model: BERTopic) -> Dict[str, float]:
    """
    计算主题多样性指标。
    
    Args:
        topic_model: 训练好的BERTopic模型
        
    Returns:
        包含多样性指标的字典
    """
    try:
        topic_info = topic_model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]
        
        if len(valid_topics) == 0:
            return {'topic_count': 0, 'avg_keywords_per_topic': 0, 'diversity_score': 0}
        
        # 计算每个主题的关键词数量
        keyword_counts = []
        all_keywords = set()
        
        for _, row in valid_topics.iterrows():
            topic_id = row['Topic']
            keywords = topic_model.get_topic(topic_id)
            keyword_words = [kw[0] for kw in keywords]
            keyword_counts.append(len(keyword_words))
            all_keywords.update(keyword_words)
        
        # 计算多样性指标
        topic_count = len(valid_topics)
        avg_keywords_per_topic = np.mean(keyword_counts)
        diversity_score = len(all_keywords) / sum(keyword_counts) if sum(keyword_counts) > 0 else 0
        
        return {
            'topic_count': topic_count,
            'avg_keywords_per_topic': avg_keywords_per_topic,
            'diversity_score': diversity_score
        }
        
    except Exception as e:
        logger.error(f"计算主题多样性失败: {e}")
        return {'topic_count': 0, 'avg_keywords_per_topic': 0, 'diversity_score': 0}


def optimize_umap_parameters(
    embeddings: np.ndarray,
    n_neighbors_range: List[int] = [5, 10, 15, 20, 30, 50],
    min_dist_range: List[float] = [0.05, 0.1, 0.2, 0.5, 0.8]
) -> Dict[str, Any]:
    """
    优化UMAP参数以获得最佳的降维效果。
    
    Args:
        embeddings: 嵌入向量
        n_neighbors_range: n_neighbors参数范围
        min_dist_range: min_dist参数范围
        
    Returns:
        最佳参数和对应的评分
    """
    from sklearn.metrics import silhouette_score
    
    best_score = -1
    best_params = {}
    best_reduced = None
    
    logger.info("开始UMAP参数优化")
    
    for n_neighbors in tqdm(n_neighbors_range, desc="优化n_neighbors"):
        for min_dist in tqdm(min_dist_range, desc="优化min_dist", leave=False):
            try:
                reduced = perform_umap(
                    embeddings,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=2
                )
                
                # 计算轮廓系数
                if len(reduced) > 2:  # 轮廓系数需要至少3个样本
                    score = silhouette_score(reduced, np.arange(len(reduced)))
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'score': score
                    }
                    best_reduced = reduced
                    
            except Exception as e:
                logger.warning(f"参数组合 n_neighbors={n_neighbors}, min_dist={min_dist} 失败: {e}")
                continue
    
    logger.info(f"UMAP参数优化完成，最佳参数: {best_params}")
    
    return {
        'best_params': best_params,
        'best_reduced_embeddings': best_reduced
    }


class ModelLoadError(Exception):
    """模型加载异常类"""
    pass


class EmbeddingError(Exception):
    """嵌入生成异常类"""
    pass


class DimensionalityReductionError(Exception):
    """降维异常类"""
    pass


class TopicModelError(Exception):
    """主题建模异常类"""
    pass


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    获取模型的详细信息。
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型信息字典
    """
    model_info = {
        # Sentence Transformers 模型
        "all-MiniLM-L6-v2": {
            "description": "轻量级通用句子嵌入模型",
            "dimensions": 384,
            "max_length": 256,
            "speed": "快",
            "quality": "中等",
            "recommended_for": "快速分析、大数据集",
            "type": "sentence-transformers"
        },
        "all-mpnet-base-v2": {
            "description": "平衡型句子嵌入模型",
            "dimensions": 768,
            "max_length": 384,
            "speed": "中等",
            "quality": "高",
            "recommended_for": "一般用途分析",
            "type": "sentence-transformers"
        },
        "text-embedding-ada-002": {
            "description": "高质量句子嵌入模型",
            "dimensions": 1536,
            "max_length": 8192,
            "speed": "慢",
            "quality": "很高",
            "recommended_for": "深度分析、高精度要求",
            "type": "sentence-transformers"
        },
        
        # Gensim 模型
        "doc2vec": {
            "description": "文档到向量嵌入模型，直接学习文档表示",
            "dimensions": 100,
            "max_length": "无限制",
            "speed": "中等",
            "quality": "中等",
            "recommended_for": "文档相似性、主题分析",
            "type": "gensim",
            "training_required": True
        },
        "word2vec": {
            "description": "词到向量嵌入模型，通过词向量聚合生成文档向量",
            "dimensions": 100,
            "max_length": "无限制",
            "speed": "快",
            "quality": "中等",
            "recommended_for": "词相似性、快速文档分析",
            "type": "gensim",
            "training_required": True
        },
        "fasttext": {
            "description": "基于Word2Vec的改进模型，支持子词信息",
            "dimensions": 100,
            "max_length": "无限制",
            "speed": "快",
            "quality": "中高",
            "recommended_for": "处理罕见词、多语言文本",
            "type": "gensim",
            "training_required": True
        },
        
        # Hugging Face Transformers 模型
        "bert-base-uncased": {
            "description": "BERT基础模型，双向上下文理解",
            "dimensions": 768,
            "max_length": 512,
            "speed": "慢",
            "quality": "很高",
            "recommended_for": "深度语义理解、复杂NLP任务",
            "type": "transformers"
        },
        "roberta-base": {
            "description": "RoBERTa基础模型，优化的BERT变体",
            "dimensions": 768,
            "max_length": 512,
            "speed": "慢",
            "quality": "很高",
            "recommended_for": "高精度文本分析、学术文献处理",
            "type": "transformers"
        },
        "distilbert-base-uncased": {
            "description": "DistilBERT基础模型，轻量级BERT",
            "dimensions": 768,
            "max_length": 512,
            "speed": "中等",
            "quality": "高",
            "recommended_for": "平衡性能和效率的应用",
            "type": "transformers"
        }
    }
    
    return model_info.get(model_name, {})


# ========== 新增主题分析模型 ==========

def perform_lda_topic_modeling(
    texts: List[str],
    n_topics: int = 10,
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.8,
    random_state: int = 42,
    alpha: float = 0.1,
    beta: float = 0.01
) -> Tuple[LatentDirichletAllocation, pd.DataFrame]:
    """
    使用LDA (Latent Dirichlet Allocation) 进行主题建模。
    
    Args:
        texts: 预处理后的文本列表
        n_topics: 主题数量
        max_features: 最大特征数量
        min_df: 最小文档频率
        max_df: 最大文档频率
        random_state: 随机种子
        alpha: 文档-主题分布的先验参数
        beta: 主题-词分布的先验参数
        
    Returns:
        (trained_model, topics_df): 训练好的LDA模型和主题信息DataFrame
    """
    if not texts:
        raise ValueError("输入文本列表不能为空")
    
    if n_topics < 2:
        raise ValueError("主题数量必须大于等于2")
    
    try:
        logger.info(f"开始LDA主题建模: n_topics={n_topics}")
        
        # 创建文档-词项矩阵
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # 创建LDA模型
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            doc_topic_prior=alpha,
            topic_word_prior=beta,
            random_state=random_state,
            max_iter=20,
            learning_method='online',
            batch_size=128,
            evaluate_every=-1,
            verbose=1
        )
        
        # 训练模型
        lda_model.fit(doc_term_matrix)
        
        # 获取主题分配
        topic_distributions = lda_model.transform(doc_term_matrix)
        topics = np.argmax(topic_distributions, axis=1)
        
        # 创建主题信息DataFrame
        topics_df = _create_classical_topics_dataframe(
            lda_model, topics, topic_distributions, vectorizer, "LDA"
        )
        
        logger.info(f"LDA主题建模完成，发现 {n_topics} 个主题")
        
        return lda_model, topics_df
        
    except Exception as e:
        logger.error(f"LDA主题建模失败: {e}")
        raise TopicModelError(f"LDA主题建模失败: {e}") from e


def perform_nmf_topic_modeling(
    texts: List[str],
    n_topics: int = 10,
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.8,
    random_state: int = 42,
    alpha: float = 0.1,
    l1_ratio: float = 0.5
) -> Tuple[NMF, pd.DataFrame]:
    """
    使用NMF (Non-negative Matrix Factorization) 进行主题建模。
    
    Args:
        texts: 预处理后的文本列表
        n_topics: 主题数量
        max_features: 最大特征数量
        min_df: 最小文档频率
        max_df: 最大文档频率
        random_state: 随机种子
        alpha: 正则化参数
        l1_ratio: L1正则化比例
        
    Returns:
        (trained_model, topics_df): 训练好的NMF模型和主题信息DataFrame
    """
    if not texts:
        raise ValueError("输入文本列表不能为空")
    
    if n_topics < 2:
        raise ValueError("主题数量必须大于等于2")
    
    try:
        logger.info(f"开始NMF主题建模: n_topics={n_topics}")
        
        # 创建TF-IDF矩阵
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        # 创建NMF模型
        nmf_model = NMF(
            n_components=n_topics,
            random_state=random_state,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=200,
            verbose=1
        )
        
        # 训练模型
        nmf_model.fit(tfidf_matrix)
        
        # 获取主题分配
        topic_distributions = nmf_model.transform(tfidf_matrix)
        topics = np.argmax(topic_distributions, axis=1)
        
        # 创建主题信息DataFrame
        topics_df = _create_classical_topics_dataframe(
            nmf_model, topics, topic_distributions, tfidf_vectorizer, "NMF"
        )
        
        logger.info(f"NMF主题建模完成，发现 {n_topics} 个主题")
        
        return nmf_model, topics_df
        
    except Exception as e:
        logger.error(f"NMF主题建模失败: {e}")
        raise TopicModelError(f"NMF主题建模失败: {e}") from e


def perform_kmeans_topic_modeling(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
    algorithm: str = 'auto',
    batch_size: Optional[int] = None
) -> Tuple[Union[KMeans, MiniBatchKMeans], pd.DataFrame]:
    """
    使用K-means聚类进行主题建模。
    
    Args:
        embeddings: 文本嵌入向量
        n_clusters: 聚类数量
        random_state: 随机种子
        algorithm: K-means算法类型
        batch_size: 批处理大小，如果指定则使用MiniBatchKMeans
        
    Returns:
        (trained_model, topics_df): 训练好的K-means模型和主题信息DataFrame
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("嵌入向量不能为空")
    
    if n_clusters < 2:
        raise ValueError("聚类数量必须大于等于2")
    
    if n_clusters > len(embeddings):
        raise ValueError("聚类数量不能超过样本数量")
    
    try:
        logger.info(f"开始K-means主题建模: n_clusters={n_clusters}")
        
        # 标准化嵌入向量
        normalized_embeddings = normalize(embeddings, norm='l2')
        
        # 创建K-means模型
        if batch_size is not None:
            # 使用MiniBatchKMeans处理大数据集
            kmeans_model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=batch_size,
                verbose=1
            )
        else:
            # 使用标准K-means
            kmeans_model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                algorithm=algorithm,
                verbose=1
            )
        
        # 训练模型
        kmeans_model.fit(normalized_embeddings)
        
        # 获取聚类标签和距离
        topics = kmeans_model.labels_
        distances = kmeans_model.transform(normalized_embeddings)
        
        # 计算主题概率（基于距离的倒数）
        topic_probabilities = 1.0 / (1.0 + distances)
        topic_probabilities = topic_probabilities / topic_probabilities.sum(axis=1, keepdims=True)
        
        # 创建主题信息DataFrame
        topics_df = _create_clustering_topics_dataframe(
            kmeans_model, topics, topic_probabilities, "KMeans"
        )
        
        logger.info(f"K-means主题建模完成，发现 {n_clusters} 个主题")
        
        return kmeans_model, topics_df
        
    except Exception as e:
        logger.error(f"K-means主题建模失败: {e}")
        raise TopicModelError(f"K-means主题建模失败: {e}") from e


def perform_hdbscan_topic_modeling(
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    metric: str = 'euclidean',
    cluster_selection_method: str = 'eom'
) -> Tuple[hdbscan.HDBSCAN, pd.DataFrame]:
    """
    使用HDBSCAN进行层次聚类主题建模。
    
    Args:
        embeddings: 文本嵌入向量
        min_cluster_size: 最小聚类大小
        min_samples: 最小样本数
        metric: 距离度量
        cluster_selection_method: 聚类选择方法
        
    Returns:
        (trained_model, topics_df): 训练好的HDBSCAN模型和主题信息DataFrame
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("嵌入向量不能为空")
    
    if min_cluster_size < 2:
        raise ValueError("最小聚类大小必须大于等于2")
    
    if min_cluster_size > len(embeddings):
        raise ValueError("最小聚类大小不能超过样本数量")
    
    try:
        logger.info(f"开始HDBSCAN主题建模: min_cluster_size={min_cluster_size}")
        
        # 标准化嵌入向量
        normalized_embeddings = normalize(embeddings, norm='l2')
        
        # 创建HDBSCAN模型
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=True
        )
        
        # 训练模型
        cluster_labels = hdbscan_model.fit_predict(normalized_embeddings)
        
        # HDBSCAN返回-1表示噪声点
        topics = cluster_labels
        
        # 计算聚类强度（作为主题概率的代理）
        strengths = hdbscan_model.probabilities_
        
        # 创建主题信息DataFrame
        topics_df = _create_clustering_topics_dataframe(
            hdbscan_model, topics, strengths, "HDBSCAN"
        )
        
        n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
        logger.info(f"HDBSCAN主题建模完成，发现 {n_clusters} 个主题")
        
        return hdbscan_model, topics_df
        
    except Exception as e:
        logger.error(f"HDBSCAN主题建模失败: {e}")
        raise TopicModelError(f"HDBSCAN主题建模失败: {e}") from e


def _create_classical_topics_dataframe(
    model: Union[LatentDirichletAllocation, NMF],
    topics: np.ndarray,
    topic_distributions: np.ndarray,
    vectorizer: Union[CountVectorizer, TfidfVectorizer],
    model_name: str
) -> pd.DataFrame:
    """
    为经典主题模型（LDA、NMF）创建主题信息DataFrame。
    
    Args:
        model: 训练好的主题模型
        topics: 主题分配数组
        topic_distributions: 主题分布矩阵
        vectorizer: 向量化器
        model_name: 模型名称
        
    Returns:
        主题信息DataFrame
    """
    # 获取特征名称
    feature_names = vectorizer.get_feature_names_out()
    
    # 为每个主题生成关键词和名称
    topic_names = {}
    topic_keywords = {}
    
    for topic_id in range(model.n_components):
        # 获取主题的关键词
        if hasattr(model, 'components_'):
            topic_words = model.components_[topic_id]
            top_indices = topic_words.argsort()[-10:][::-1]
            keywords = [(feature_names[i], topic_words[i]) for i in top_indices]
        else:
            keywords = []
        
        topic_keywords[topic_id] = keywords
        
        # 生成主题名称
        if keywords:
            top_words = [kw[0] for kw in keywords[:3]]
            topic_names[topic_id] = f"{model_name}_{'_'.join(top_words)}"
        else:
            topic_names[topic_id] = f"{model_name}_Topic_{topic_id}"
    
    # 添加未分类主题
    topic_names[-1] = "Unclassified"
    
    # 创建DataFrame
    topics_data = []
    for i, (topic_id, distribution) in enumerate(zip(topics, topic_distributions)):
        # 安全地获取主题概率
        if topic_id != -1 and topic_id < len(distribution):
            topic_probability = float(distribution[topic_id])
        else:
            topic_probability = 0.0
        
        topic_data = {
            'document_id': i,
            'topic_id': topic_id,
            'topic_name': topic_names.get(topic_id, "Unknown"),
            'topic_probability': topic_probability
        }
        topics_data.append(topic_data)
    
    topics_df = pd.DataFrame(topics_data)
    
    # 存储主题关键词信息
    topics_df.attrs['topic_keywords'] = topic_keywords
    topics_df.attrs['model_name'] = model_name
    
    logger.info(f"经典主题模型DataFrame创建完成，包含 {len(topics_df)} 条记录")
    
    return topics_df


def _create_clustering_topics_dataframe(
    model: Union[KMeans, MiniBatchKMeans, hdbscan.HDBSCAN],
    topics: np.ndarray,
    strengths: np.ndarray,
    model_name: str
) -> pd.DataFrame:
    """
    为聚类主题模型（K-means、HDBSCAN）创建主题信息DataFrame。
    
    Args:
        model: 训练好的聚类模型
        topics: 聚类标签数组
        strengths: 聚类强度/概率数组
        model_name: 模型名称
        
    Returns:
        主题信息DataFrame
    """
    # 生成主题名称
    unique_topics = set(topics)
    topic_names = {}
    
    for topic_id in unique_topics:
        if topic_id == -1:
            topic_names[topic_id] = "Unclassified"
        else:
            topic_names[topic_id] = f"{model_name}_Cluster_{topic_id}"
    
    # 创建DataFrame
    topics_data = []
    for i, (topic_id, strength) in enumerate(zip(topics, strengths)):
        # 安全地获取主题强度
        if isinstance(strength, (int, float, np.number)):
            topic_strength = float(strength)
        else:
            topic_strength = 0.0
        
        topic_data = {
            'document_id': i,
            'topic_id': topic_id,
            'topic_name': topic_names.get(topic_id, "Unknown"),
            'topic_probability': topic_strength
        }
        topics_data.append(topic_data)
    
    topics_df = pd.DataFrame(topics_data)
    
    # 存储模型信息
    topics_df.attrs['model_name'] = model_name
    
    if hasattr(model, 'cluster_centers_'):
        topics_df.attrs['cluster_centers'] = model.cluster_centers_
    
    logger.info(f"聚类主题模型DataFrame创建完成，包含 {len(topics_df)} 条记录")
    
    return topics_df


def get_classical_topic_keywords(topics_df: pd.DataFrame, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    获取经典主题模型（LDA、NMF）的主题关键词。
    
    Args:
        topics_df: 主题信息DataFrame
        topic_id: 主题ID
        top_n: 返回的关键词数量
        
    Returns:
        关键词及其权重的列表
    """
    if topic_id == -1:
        return []
    
    if 'topic_keywords' not in topics_df.attrs:
        return []
    
    topic_keywords = topics_df.attrs['topic_keywords']
    keywords = topic_keywords.get(topic_id, [])
    
    return keywords[:top_n]


def get_clustering_topic_info(topics_df: pd.DataFrame, topic_id: int) -> Dict[str, Any]:
    """
    获取聚类主题模型（K-means、HDBSCAN）的主题信息。
    
    Args:
        topics_df: 主题信息DataFrame
        topic_id: 主题ID
        
    Returns:
        主题信息字典
    """
    if topic_id == -1:
        return {'cluster_id': -1, 'name': 'Unclassified'}
    
    topic_info = {
        'cluster_id': topic_id,
        'name': f"{topics_df.attrs.get('model_name', 'Cluster')}_Cluster_{topic_id}"
    }
    
    # 如果有聚类中心信息，添加到结果中
    if 'cluster_centers' in topics_df.attrs and topic_id < len(topics_df.attrs['cluster_centers']):
        topic_info['cluster_center'] = topics_df.attrs['cluster_centers'][topic_id]
    
    return topic_info


def calculate_classical_topic_diversity(topics_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算经典主题模型的多样性指标。
    
    Args:
        topics_df: 主题信息DataFrame
        
    Returns:
        包含多样性指标的字典
    """
    try:
        # 过滤有效主题
        valid_topics = topics_df[topics_df['topic_id'] != -1]
        
        if len(valid_topics) == 0:
            return {'topic_count': 0, 'avg_documents_per_topic': 0, 'diversity_score': 0}
        
        # 统计主题分布
        topic_counts = valid_topics['topic_id'].value_counts()
        
        # 计算多样性指标
        topic_count = len(topic_counts)
        avg_documents_per_topic = topic_counts.mean()
        
        # 计算香农熵作为多样性指标
        probabilities = topic_counts / topic_counts.sum()
        shannon_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        diversity_score = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'topic_count': topic_count,
            'avg_documents_per_topic': avg_documents_per_topic,
            'diversity_score': diversity_score,
            'shannon_entropy': shannon_entropy
        }
        
    except Exception as e:
        logger.error(f"计算经典主题多样性失败: {e}")
        return {'topic_count': 0, 'avg_documents_per_topic': 0, 'diversity_score': 0}


def perform_topic_modeling_ensemble(
    texts: List[str],
    embeddings: np.ndarray,
    models: List[str] = ['BERTopic', 'LDA', 'NMF', 'KMeans', 'HDBSCAN'],
    **kwargs
) -> Dict[str, Tuple[Any, pd.DataFrame]]:
    """
    执行集成主题建模，同时运行多种主题模型。
    
    Args:
        texts: 预处理后的文本列表
        embeddings: 文本嵌入向量
        models: 要使用的模型列表
        **kwargs: 各模型的参数
        
    Returns:
        字典：{模型名称: (trained_model, topics_df)}
    """
    results = {}
    
    logger.info(f"开始集成主题建模，使用模型: {models}")
    
    for model_name in models:
        try:
            if model_name == 'BERTopic':
                model_params = kwargs.get('BERTopic', {})
                result = perform_topic_modeling(texts, embeddings, **model_params)
            
            elif model_name == 'LDA':
                model_params = kwargs.get('LDA', {})
                result = perform_lda_topic_modeling(texts, **model_params)
            
            elif model_name == 'NMF':
                model_params = kwargs.get('NMF', {})
                result = perform_nmf_topic_modeling(texts, **model_params)
            
            elif model_name == 'KMeans':
                model_params = kwargs.get('KMeans', {})
                result = perform_kmeans_topic_modeling(embeddings, **model_params)
            
            elif model_name == 'HDBSCAN':
                model_params = kwargs.get('HDBSCAN', {})
                result = perform_hdbscan_topic_modeling(embeddings, **model_params)
            
            else:
                logger.warning(f"未知的主题模型: {model_name}")
                continue
            
            results[model_name] = result
            logger.info(f"{model_name} 主题建模完成")
            
        except Exception as e:
            logger.error(f"{model_name} 主题建模失败: {e}")
            continue
    
    logger.info(f"集成主题建模完成，成功运行 {len(results)} 个模型")
    
    return results


def compare_topic_models(
    model_results: Dict[str, Tuple[Any, pd.DataFrame]]
) -> pd.DataFrame:
    """
    比较不同主题模型的结果。
    
    Args:
        model_results: 模型结果字典
        
    Returns:
        包含比较结果的DataFrame
    """
    comparison_data = []
    
    for model_name, (model, topics_df) in model_results.items():
        # 计算各项指标
        if model_name in ['LDA', 'NMF']:
            diversity_metrics = calculate_classical_topic_diversity(topics_df)
        elif model_name == 'BERTopic':
            # 假设model是BERTopic模型
            try:
                diversity_metrics = calculate_topic_diversity(model)
            except:
                diversity_metrics = {'topic_count': 0, 'avg_keywords_per_topic': 0, 'diversity_score': 0}
        else:  # KMeans, HDBSCAN
            diversity_metrics = calculate_classical_topic_diversity(topics_df)
        
        # 计算文档覆盖率（非未分类文档的比例）
        total_docs = len(topics_df)
        classified_docs = len(topics_df[topics_df['topic_id'] != -1])
        coverage_rate = classified_docs / total_docs if total_docs > 0 else 0
        
        comparison_data.append({
            'model': model_name,
            'topic_count': diversity_metrics.get('topic_count', 0),
            'document_coverage': coverage_rate,
            'diversity_score': diversity_metrics.get('diversity_score', 0),
            'avg_documents_per_topic': diversity_metrics.get('avg_documents_per_topic', 0) or 
                                      diversity_metrics.get('avg_keywords_per_topic', 0),
            'total_documents': total_docs,
            'classified_documents': classified_docs
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    logger.info(f"主题模型比较完成，比较了 {len(comparison_df)} 个模型")
    
    return comparison_df


# 可用的主题模型
AVAILABLE_TOPIC_MODELS = {
    'BERTopic': {
        'description': '基于嵌入的主题建模，结合UMAP降维和HDBSCAN聚类',
        'requires_embeddings': True,
        'requires_texts': True,
        'auto_topic_count': True
    },
    'LDA': {
        'description': '经典的潜在狄利克雷分配主题模型',
        'requires_embeddings': False,
        'requires_texts': True,
        'auto_topic_count': False
    },
    'NMF': {
        'description': '非负矩阵分解主题模型，基于TF-IDF',
        'requires_embeddings': False,
        'requires_texts': True,
        'auto_topic_count': False
    },
    'KMeans': {
        'description': 'K-means聚类主题模型，基于嵌入向量',
        'requires_embeddings': True,
        'requires_texts': False,
        'auto_topic_count': False
    },
    'HDBSCAN': {
        'description': '基于密度的层次聚类主题模型，自动确定聚类数量',
        'requires_embeddings': True,
        'requires_texts': False,
        'auto_topic_count': True
    }
}


def get_topic_model_info(model_name: str) -> Dict[str, Any]:
    """
    获取主题模型的详细信息。
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型信息字典
    """
    return AVAILABLE_TOPIC_MODELS.get(model_name, {})


# ========== 新增AI集成功能 ==========

def perform_semantic_search(
    query_text: str, 
    df: pd.DataFrame, 
    embedding_model, 
    text_column: str = 'full_text',
    top_k: int = 5
) -> pd.DataFrame:
    """
    执行语义搜索，找到与查询最相似的文献。
    
    Args:
        query_text: 查询文本
        df: 包含文献数据的DataFrame
        embedding_model: 嵌入模型
        text_column: 用于搜索的文本列名
        top_k: 返回的结果数量
        
    Returns:
        包含搜索结果的DataFrame，按相似度排序
    """
    try:
        logger.info(f"开始语义搜索: '{query_text}'")
        
        # 生成查询的嵌入向量
        query_embedding = embedding_model.encode([query_text])[0]
        
        # 获取所有文献的嵌入向量
        if 'embeddings' not in df.columns:
            logger.warning("DataFrame中没有找到嵌入向量列，正在生成...")
            # 如果没有预计算的嵌入，则实时生成
            text_embeddings = embedding_model.encode(df[text_column].tolist())
            df['embeddings'] = list(text_embeddings)
        
        # 计算余弦相似度
        similarities = []
        for idx, row in df.iterrows():
            doc_embedding = np.array(row['embeddings'])
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # 将相似度添加到DataFrame
        df_search = df.copy()
        df_search['similarity'] = similarities
        
        # 按相似度排序并返回前k个结果
        result = df_search.nlargest(top_k, 'similarity')
        
        logger.info(f"语义搜索完成，找到 {len(result)} 个相关文献")
        
        return result[['article_title', 'journal_title', 'publication_year', 'similarity', text_column]]
        
    except Exception as e:
        logger.error(f"语义搜索失败: {e}")
        raise RuntimeError(f"语义搜索失败: {e}") from e


def answer_question_with_llm(
    question: str, 
    context_texts: List[str], 
    llm_client: LLMClient,
    max_context_length: int = 3000
) -> str:
    """
    使用LLM基于提供的上下文回答问题。
    
    Args:
        question: 用户问题
        context_texts: 上下文文本列表
        llm_client: LLM客户端实例
        max_context_length: 上下文的最大长度
        
    Returns:
        LLM生成的答案
    """
    try:
        logger.info(f"开始使用LLM回答问题: '{question}'")
        
        # 准备上下文
        context = "\n\n".join(context_texts)
        
        # 如果上下文太长，进行截断
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # 构建prompt
        prompt = f"""请基于以下提供的文献摘要信息回答用户的问题。请确保你的回答基于提供的信息，如果信息不足，请明确说明。

文献摘要信息：
{context}

用户问题：{question}

请提供详细、准确的回答："""
        
        # 调用LLM生成答案
        answer = llm_client.generate_text(prompt, temperature=0.3, max_tokens=800)
        
        logger.info("LLM回答生成完成")
        
        return answer
        
    except Exception as e:
        logger.error(f"LLM回答生成失败: {e}")
        return f"很抱歉，生成答案时遇到错误: {e}"


def generate_llm_topic_label(
    topic_keywords: List[Tuple[str, float]], 
    representative_docs_snippets: List[str], 
    llm_client: LLMClient,
    max_length: int = 50
) -> str:
    """
    使用LLM为主题生成更准确的标签。
    
    Args:
        topic_keywords: 主题关键词及其权重的列表
        representative_docs_snippets: 代表性文档片段列表
        llm_client: LLM客户端实例
        max_length: 标签的最大长度
        
    Returns:
        LLM生成的主题标签
    """
    try:
        logger.info("开始使用LLM生成主题标签")
        
        # 准备关键词信息
        keywords_text = ", ".join([f"{word}({weight:.2f})" for word, weight in topic_keywords[:10]])
        
        # 准备文档片段
        docs_text = "\n".join([f"- {doc[:200]}..." for doc in representative_docs_snippets[:5]])
        
        # 构建prompt
        prompt = f"""基于以下主题的关键词和代表性文档片段，请生成一个简洁、准确的主题名称（不超过{max_length}个字符）。

关键词及权重：
{keywords_text}

代表性文档片段：
{docs_text}

请生成一个能够准确反映这个主题核心内容的名称："""
        
        # 调用LLM生成标签
        label = llm_client.generate_text(prompt, temperature=0.5, max_tokens=100)
        
        # 清理和截断标签
        label = label.strip().strip('"\'').strip()
        if len(label) > max_length:
            label = label[:max_length-3] + "..."
        
        logger.info(f"LLM主题标签生成完成: '{label}'")
        
        return label
        
    except Exception as e:
        logger.error(f"LLM主题标签生成失败: {e}")
        return "主题标签生成失败"


def get_temporal_distribution(
    df: pd.DataFrame, 
    entity_col: str,
    year_col: str = 'publication_year'
) -> pd.DataFrame:
    """
    获取实体（主题、期刊等）的时间分布数据。
    
    Args:
        df: 包含数据的DataFrame
        entity_col: 实体列名（如'topic_name', 'journal_title'）
        year_col: 年份列名
        
    Returns:
        包含时间分布数据的DataFrame
    """
    try:
        logger.info(f"开始计算{entity_col}的时间分布")
        
        # 按年份和实体分组计数
        temporal_data = df.groupby([year_col, entity_col]).size().reset_index(name='count')
        
        # 计算每年的总数
        yearly_totals = df.groupby(year_col).size().reset_index(name='yearly_total')
        
        # 合并数据
        temporal_data = temporal_data.merge(yearly_totals, on=year_col)
        
        # 计算比例
        temporal_data['proportion'] = temporal_data['count'] / temporal_data['yearly_total']
        
        # 按年份排序
        temporal_data = temporal_data.sort_values(year_col)
        
        logger.info(f"时间分布计算完成，包含 {len(temporal_data)} 条记录")
        
        return temporal_data
        
    except Exception as e:
        logger.error(f"时间分布计算失败: {e}")
        raise RuntimeError(f"时间分布计算失败: {e}") from e


def extract_keywords_advanced(
    texts: List[str],
    method: str = 'yake',
    max_keywords: int = 20,
    language: str = 'en'
) -> List[Tuple[str, float]]:
    """
    使用高级算法提取关键词。
    
    Args:
        texts: 文本列表
        method: 关键词提取方法 ('yake', 'keybert', 'tfidf')
        max_keywords: 最大关键词数量
        language: 语言
        
    Returns:
        关键词及其权重的列表
    """
    try:
        logger.info(f"开始使用{method}方法提取关键词")
        
        if method == 'yake':
            try:
                import yake
                kw_extractor = yake.KeywordExtractor(
                    lan=language,
                    n=3,  # n-gram大小
                    dedupLim=0.9,  # 去重限制
                    top=max_keywords
                )
                
                # 合并所有文本
                combined_text = " ".join(texts)
                keywords = kw_extractor.extract_keywords(combined_text)
                
                # YAKE返回的是(关键词, 分数)，分数越低越好，我们需要转换为权重
                keywords = [(kw, 1.0 - score) for kw, score in keywords]
                
            except ImportError:
                logger.warning("YAKE未安装，回退到TF-IDF方法")
                method = 'tfidf'
        
        if method == 'keybert':
            try:
                from keybert import KeyBERT
                kw_model = KeyBERT()
                
                # 合并所有文本
                combined_text = " ".join(texts)
                keywords = kw_model.extract_keywords(
                    combined_text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    use_maxsum=True,
                    nr_candidates=max_keywords,
                    top_n=max_keywords
                )
                
            except ImportError:
                logger.warning("KeyBERT未安装，回退到TF-IDF方法")
                method = 'tfidf'
        
        if method == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(
                max_features=max_keywords,
                stop_words='english',
                ngram_range=(1, 3)
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算每个词的平均TF-IDF分数
            mean_scores = tfidf_matrix.mean(axis=0).A1
            keywords = list(zip(feature_names, mean_scores))
            keywords.sort(key=lambda x: x[1], reverse=True)
            keywords = keywords[:max_keywords]
        
        logger.info(f"关键词提取完成，提取了 {len(keywords)} 个关键词")
        
        return keywords
        
    except Exception as e:
        logger.error(f"关键词提取失败: {e}")
        return []


def build_cooccurrence_matrix(
    df: pd.DataFrame,
    entity_col: str,
    text_col: str = 'processed_text',
    min_cooccurrence: int = 2
) -> pd.DataFrame:
    """
    构建实体共现矩阵。
    
    Args:
        df: 包含数据的DataFrame
        entity_col: 实体列名
        text_col: 文本列名
        min_cooccurrence: 最小共现次数
        
    Returns:
        共现矩阵DataFrame
    """
    try:
        logger.info(f"开始构建{entity_col}的共现矩阵")
        
        # 获取所有唯一实体
        entities = df[entity_col].unique()
        
        # 初始化共现矩阵
        cooccurrence_matrix = pd.DataFrame(
            0, 
            index=entities, 
            columns=entities
        )
        
        # 计算共现次数
        for _, row in df.iterrows():
            current_entity = row[entity_col]
            text = row[text_col]
            
            # 找出在同一文本中出现的其他实体
            cooccurring_entities = df[
                (df[text_col] == text) & 
                (df[entity_col] != current_entity)
            ][entity_col].unique()
            
            # 更新共现矩阵
            for other_entity in cooccurring_entities:
                cooccurrence_matrix.loc[current_entity, other_entity] += 1
                cooccurrence_matrix.loc[other_entity, current_entity] += 1  # 对称矩阵
        
        # 过滤低频共现
        cooccurrence_matrix = cooccurrence_matrix.mask(
            cooccurrence_matrix < min_cooccurrence, 0
        )
        
        logger.info(f"共现矩阵构建完成，形状: {cooccurrence_matrix.shape}")
        
        return cooccurrence_matrix
        
    except Exception as e:
        logger.error(f"共现矩阵构建失败: {e}")
        raise RuntimeError(f"共现矩阵构建失败: {e}") from e


def perform_topic_modeling_with_factory(
    texts: List[str],
    embeddings: Optional[np.ndarray] = None,
    model_type: str = "BERTopic",
    **kwargs
) -> Tuple[AbstractTopicModel, pd.DataFrame]:
    """
    使用主题模型工厂执行主题建模。
    
    Args:
        texts: 文本列表
        embeddings: 嵌入向量（可选）
        model_type: 主题模型类型
        **kwargs: 模型参数
        
    Returns:
        (trained_model, topics_df): 训练好的主题模型和主题信息DataFrame
    """
    try:
        logger.info(f"开始使用{model_type}进行主题建模")
        
        # 使用工厂创建模型
        topic_model = TopicModelFactory.create_model(model_type, **kwargs)
        
        # 训练模型
        topics = topic_model.fit_transform(texts, embeddings)
        
        # 获取主题信息
        topic_info = topic_model.get_topic_info()
        
        # 创建包含主题信息的DataFrame
        topics_data = []
        for i, topic_id in enumerate(topics):
            # 获取主题名称
            if topic_id == -1:
                topic_name = "Unclassified"
            else:
                topic_keywords = topic_model.get_topic(topic_id)
                if topic_keywords:
                    top_words = [kw[0] for kw in topic_keywords[:3]]
                    topic_name = f"{model_type}_{'_'.join(top_words)}"
                else:
                    topic_name = f"{model_type}_Topic_{topic_id}"
            
            topic_data = {
                'document_id': i,
                'topic_id': topic_id,
                'topic_name': topic_name
            }
            topics_data.append(topic_data)
        
        topics_df = pd.DataFrame(topics_data)
        
        logger.info(f"主题建模完成，发现 {len(set(topics)) - (1 if -1 in topics else 0)} 个主题")
        
        return topic_model, topics_df
        
    except Exception as e:
        logger.error(f"主题建模失败: {e}")
        raise TopicModelError(f"主题建模失败: {e}") from e
