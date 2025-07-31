from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Tuple, Optional
import numpy as np

class AbstractTopicModel(ABC):
    @abstractmethod
    def fit_transform(self, documents: List[str], embeddings: Optional[List] = None) -> List[int]:
        """训练模型并返回文档的主题分配。"""
        pass

    @abstractmethod
    def get_topic_info(self) -> pd.DataFrame:
        """获取主题的详细信息（ID, 名称, 数量等）。"""
        pass

    @abstractmethod
    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        """获取指定主题的关键词。"""
        pass

    @abstractmethod
    def get_representative_docs(self, topic_id: int) -> List[int]:
        """获取指定主题的代表性文档索引。"""
        pass

    @abstractmethod
    def get_document_topics(self) -> List[int]:
        """获取每个文档的主题ID。"""
        pass


class BERTopicWrapper(AbstractTopicModel):
    def __init__(self, **kwargs):
        try:
            from bertopic import BERTopic
        except ImportError:
            raise ImportError("BERTopic not installed. Please install with: pip install bertopic")
        
        self.model = BERTopic(**kwargs)
        self.topics_ = None
        self.probabilities_ = None
        self.documents_ = None

    def fit_transform(self, documents: List[str], embeddings: Optional[List] = None) -> List[int]:
        self.documents_ = documents
        if embeddings:
            self.topics_, self.probabilities_ = self.model.fit_transform(documents, embeddings)
        else:
            self.topics_, self.probabilities_ = self.model.fit_transform(documents)
        return self.topics_

    def get_topic_info(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        return self.model.get_topic_info()

    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        if self.model is None:
            return []
        return self.model.get_topic(topic_id)

    def get_representative_docs(self, topic_id: int) -> List[int]:
        if self.model is None or self.documents_ is None:
            return []
        
        # 获取该主题的文档索引
        topic_docs_indices = [i for i, topic in enumerate(self.topics_) if topic == topic_id]
        
        # 如果没有概率信息，返回所有相关文档
        if self.probabilities_ is None:
            return topic_docs_indices[:10]  # 返回前10个
        
        # 根据概率排序，返回最相关的文档
        doc_probs = [(i, self.probabilities_[i][topic_id] if topic_id < len(self.probabilities_[i]) else 0) 
                     for i in topic_docs_indices]
        doc_probs.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in doc_probs[:10]]

    def get_document_topics(self) -> List[int]:
        return self.topics_ if self.topics_ is not None else []


class LDAWrapper(AbstractTopicModel):
    def __init__(self, num_topics: int = 10, **kwargs):
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
        except ImportError:
            raise ImportError("scikit-learn not installed. Please install with: pip install scikit-learn")
        
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=num_topics, **kwargs)
        self.topics_ = None
        self.documents_ = None
        self.feature_names_ = None

    def fit_transform(self, documents: List[str], embeddings: Optional[List] = None) -> List[int]:
        self.documents_ = documents
        
        # 向量化文档
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        
        # 训练LDA模型
        doc_topic_matrix = self.lda_model.fit_transform(doc_term_matrix)
        
        # 获取每个文档的主要主题
        self.topics_ = np.argmax(doc_topic_matrix, axis=1).tolist()
        
        return self.topics_

    def get_topic_info(self) -> pd.DataFrame:
        if self.lda_model is None:
            return pd.DataFrame()
        
        topic_info = []
        for topic_id in range(self.num_topics):
            topic_words = self.get_topic(topic_id)
            topic_name = f"Topic {topic_id}: " + ", ".join([word for word, _ in topic_words[:5]])
            
            # 计算每个主题的文档数量
            doc_count = sum(1 for topic in self.topics_ if topic == topic_id)
            
            topic_info.append({
                'Topic': topic_id,
                'Name': topic_name,
                'Count': doc_count
            })
        
        return pd.DataFrame(topic_info)

    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        if self.lda_model is None or topic_id >= self.num_topics:
            return []
        
        # 获取主题词分布
        topic_word_dist = self.lda_model.components_[topic_id]
        
        # 获取最重要的词
        top_words_idx = topic_word_dist.argsort()[-10:][::-1]
        top_words = [(self.feature_names_[i], topic_word_dist[i]) for i in top_words_idx]
        
        return top_words

    def get_representative_docs(self, topic_id: int) -> List[int]:
        if self.lda_model is None or self.documents_ is None:
            return []
        
        # 获取该主题的文档索引
        topic_docs_indices = [i for i, topic in enumerate(self.topics_) if topic == topic_id]
        
        return topic_docs_indices[:10]  # 返回前10个文档

    def get_document_topics(self) -> List[int]:
        return self.topics_ if self.topics_ is not None else []


class NMFWrapper(AbstractTopicModel):
    def __init__(self, num_topics: int = 10, **kwargs):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import NMF
        except ImportError:
            raise ImportError("scikit-learn not installed. Please install with: pip install scikit-learn")
        
        self.num_topics = num_topics
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.nmf_model = NMF(n_components=num_topics, **kwargs)
        self.topics_ = None
        self.documents_ = None
        self.feature_names_ = None

    def fit_transform(self, documents: List[str], embeddings: Optional[List] = None) -> List[int]:
        self.documents_ = documents
        
        # 向量化文档
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        
        # 训练NMF模型
        doc_topic_matrix = self.nmf_model.fit_transform(doc_term_matrix)
        
        # 获取每个文档的主要主题
        self.topics_ = np.argmax(doc_topic_matrix, axis=1).tolist()
        
        return self.topics_

    def get_topic_info(self) -> pd.DataFrame:
        if self.nmf_model is None:
            return pd.DataFrame()
        
        topic_info = []
        for topic_id in range(self.num_topics):
            topic_words = self.get_topic(topic_id)
            topic_name = f"Topic {topic_id}: " + ", ".join([word for word, _ in topic_words[:5]])
            
            # 计算每个主题的文档数量
            doc_count = sum(1 for topic in self.topics_ if topic == topic_id)
            
            topic_info.append({
                'Topic': topic_id,
                'Name': topic_name,
                'Count': doc_count
            })
        
        return pd.DataFrame(topic_info)

    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        if self.nmf_model is None or topic_id >= self.num_topics:
            return []
        
        # 获取主题词分布
        topic_word_dist = self.nmf_model.components_[topic_id]
        
        # 获取最重要的词
        top_words_idx = topic_word_dist.argsort()[-10:][::-1]
        top_words = [(self.feature_names_[i], topic_word_dist[i]) for i in top_words_idx]
        
        return top_words

    def get_representative_docs(self, topic_id: int) -> List[int]:
        if self.nmf_model is None or self.documents_ is None:
            return []
        
        # 获取该主题的文档索引
        topic_docs_indices = [i for i, topic in enumerate(self.topics_) if topic == topic_id]
        
        return topic_docs_indices[:10]  # 返回前10个文档

    def get_document_topics(self) -> List[int]:
        return self.topics_ if self.topics_ is not None else []


class TopicModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs) -> AbstractTopicModel:
        if model_type == "BERTopic":
            return BERTopicWrapper(**kwargs)
        elif model_type == "LDA":
            return LDAWrapper(**kwargs)
        elif model_type == "NMF":
            return NMFWrapper(**kwargs)
        else:
            raise ValueError(f"不支持的主题模型类型: {model_type}")
