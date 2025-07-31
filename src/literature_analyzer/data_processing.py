"""
数据处理模块

负责数据导入、合并、清洗和预处理功能。
严格按照规约对WoS Excel文件进行处理。
"""

import re
import hashlib
import logging
from typing import List, Optional, Tuple
import pandas as pd
import spacy

# 配置日志
logger = logging.getLogger(__name__)

# 加载spaCy英语模型
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy英语模型加载成功")
except OSError:
    logger.warning("spaCy英语模型未找到，正在尝试下载...")
    try:
        # 尝试自动下载模型
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy英语模型下载并加载成功")
    except Exception as e:
        logger.error(f"无法加载或下载spaCy模型: {e}")
        raise RuntimeError("无法初始化spaCy模型，请手动运行: python -m spacy download en_core_web_sm")
except Exception as e:
    logger.error(f"初始化spaCy模型失败: {e}")
    raise


def load_and_process_data(uploaded_files: List) -> pd.DataFrame:
    """
    加载并处理上传的WoS Excel文件。
    
    这是数据处理的主要入口函数，按照规约严格按序执行：
    1. 文件合并
    2. 列筛选与重命名
    3. 数据清洗
    4. 文本预处理
    
    Args:
        uploaded_files: 用户上传的文件列表
        
    Returns:
        处理后的DataFrame，包含所有必需的列
        
    Raises:
        ValueError: 当没有上传文件或数据处理失败时
        ProcessingError: 当数据处理过程中出现错误时
    """
    if not uploaded_files:
        raise ValueError("请上传至少一个WoS Excel文件")
    
    logger.info(f"开始处理 {len(uploaded_files)} 个文件")
    
    try:
        # 步骤1: 文件合并
        raw_df = _merge_files(uploaded_files)
        logger.info(f"原始数据加载完成，共 {len(raw_df)} 条记录")
        
        # 步骤2: 列筛选与重命名
        filtered_df = _filter_and_rename_columns(raw_df)
        logger.info(f"列筛选完成，保留 {len(filtered_df)} 条记录")
        
        # 步骤3: 数据清洗
        cleaned_df = _clean_data(filtered_df)
        logger.info(f"数据清洗完成，保留 {len(cleaned_df)} 条记录")
        
        # 步骤4: 文本预处理
        processed_df = _preprocess_text_data(cleaned_df)
        logger.info(f"文本预处理完成，最终保留 {len(processed_df)} 条记录")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}")
        raise ProcessingError(f"数据处理失败: {str(e)}") from e


def _merge_files(uploaded_files: List) -> pd.DataFrame:
    """
    合并多个Excel文件为一个DataFrame。
    
    Args:
        uploaded_files: 上传的文件列表
        
    Returns:
        合并后的DataFrame
    """
    dataframes = []
    
    for file in uploaded_files:
        try:
            logger.info(f"正在读取文件: {file.name}")
            df = pd.read_excel(file)
            
            if df.empty:
                logger.warning(f"文件 {file.name} 为空，跳过")
                continue
                
            dataframes.append(df)
            
        except Exception as e:
            logger.error(f"读取文件 {file.name} 失败: {e}")
            continue
    
    if not dataframes:
        raise ValueError("没有成功读取任何文件")
    
    # 合并所有DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


def _filter_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    筛选并重命名列。
    
    仅保留必需的列并重命名为内部标准名称：
    - 'Article Title' -> 'article_title'
    - 'Source Title' -> 'journal_title'
    - 'Publication Year' -> 'publication_year'
    - 'Abstract' -> 'abstract_text'
    
    Args:
        df: 输入DataFrame
        
    Returns:
        筛选重命名后的DataFrame
    """
    # 定义列映射
    column_mapping = {
        'Article Title': 'article_title',
        'Source Title': 'journal_title',
        'Publication Year': 'publication_year',
        'Abstract': 'abstract_text'
    }
    
    # 检查必需列是否存在
    missing_columns = []
    for original_col in column_mapping.keys():
        if original_col not in df.columns:
            missing_columns.append(original_col)
    
    if missing_columns:
        raise ValueError(f"Excel文件中缺少必需的列: {missing_columns}")
    
    # 筛选并重命名列
    filtered_df = df[list(column_mapping.keys())].copy()
    filtered_df = filtered_df.rename(columns=column_mapping)
    
    return filtered_df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗步骤。
    
    执行以下清洗操作：
    1. 基于article_title和abstract_text去重
    2. 处理缺失值
    3. 类型转换
    4. 内容过滤（摘要长度）
    
    Args:
        df: 输入DataFrame
        
    Returns:
        清洗后的DataFrame
    """
    cleaned_df = df.copy()
    original_count = len(cleaned_df)
    
    # 1. 去重：基于article_title和abstract_text的组合
    cleaned_df = cleaned_df.drop_duplicates(
        subset=['article_title', 'abstract_text'],
        keep='first'
    )
    logger.info(f"去重后保留 {len(cleaned_df)} 条记录")
    
    # 2. 缺失值处理：删除任何在必需列中包含缺失值的行
    required_columns = ['article_title', 'journal_title', 'publication_year', 'abstract_text']
    cleaned_df = cleaned_df.dropna(subset=required_columns)
    logger.info(f"缺失值处理后保留 {len(cleaned_df)} 条记录")
    
    # 3. 类型转换：将publication_year转换为整数
    try:
        cleaned_df['publication_year'] = cleaned_df['publication_year'].astype(int)
    except (ValueError, TypeError) as e:
        logger.error(f"年份类型转换失败: {e}")
        # 移除无法转换的行
        cleaned_df = cleaned_df[cleaned_df['publication_year'].apply(
            lambda x: str(x).isdigit()
        )]
        cleaned_df['publication_year'] = cleaned_df['publication_year'].astype(int)
        logger.info(f"年份转换后保留 {len(cleaned_df)} 条记录")
    
    # 4. 内容过滤：删除abstract_text字符数少于100的记录
    cleaned_df = cleaned_df[cleaned_df['abstract_text'].str.len() >= 100]
    logger.info(f"内容过滤后保留 {len(cleaned_df)} 条记录")
    
    final_count = len(cleaned_df)
    logger.info(f"数据清洗完成：{original_count} -> {final_count} (保留率: {final_count/original_count:.1%})")
    
    return cleaned_df


def _preprocess_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    文本预处理步骤。
    
    执行以下预处理操作：
    1. 创建full_text列（标题+摘要）
    2. 创建processed_text列，应用完整的文本预处理流程
    
    Args:
        df: 输入DataFrame
        
    Returns:
        包含预处理文本的DataFrame
    """
    processed_df = df.copy()
    
    # 1. 合并文本：创建full_text列
    processed_df['full_text'] = (
        processed_df['article_title'].fillna('') + ' ' + 
        processed_df['abstract_text'].fillna('')
    ).str.strip()
    
    # 2. 创建processed_text列
    logger.info("开始文本预处理...")
    processed_df['processed_text'] = processed_df['full_text'].apply(preprocess_text)
    logger.info("文本预处理完成")
    
    return processed_df


def preprocess_text(text: str) -> str:
    """
    对单个文本进行预处理。
    
    预处理步骤：
    1. 规范化：转换为小写，移除URL和HTML标签
    2. 使用spaCy进行分词、停用词移除、词形还原和词性标注
    3. 过滤保留有意义的词汇（名词、形容词、动词、副词）
    4. 空格处理
    
    Args:
        text: 输入文本
        
    Returns:
        预处理后的文本
        
    Raises:
        ValueError: 当输入文本为空或无效时
    """
    if not text or not isinstance(text, str):
        raise ValueError("输入文本必须是非空字符串")
    
    try:
        # 1. 规范化
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 使用spaCy处理文本
        doc = nlp(text)
        
        # 3. 提取有意义的词汇：过滤停用词、标点，保留名词、形容词、动词、副词
        # 同时进行词形还原
        meaningful_tokens = []
        for token in doc:
            # 保留有意义的词性：名词、形容词、动词、副词
            if (token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV'] and 
                not token.is_stop and 
                not token.is_punct and 
                not token.is_space and 
                len(token.text) > 2):
                # 使用词形还原（lemma）
                meaningful_tokens.append(token.lemma_)
        
        # 4. 空格处理：合并并移除多余空格
        processed_text = ' '.join(meaningful_tokens).strip()
        
        # 移除多余的空格
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        return processed_text
        
    except Exception as e:
        logger.error(f"文本预处理失败: {e}")
        raise ProcessingError(f"文本预处理失败: {e}") from e


def get_data_hash(uploaded_files: List) -> str:
    """
    计算上传文件的哈希值，用于缓存标识。
    
    Args:
        uploaded_files: 上传的文件列表
        
    Returns:
        文件内容的哈希字符串
    """
    hash_obj = hashlib.md5()
    
    for file in uploaded_files:
        # 读取文件内容并更新哈希
        file_content = file.read()
        hash_obj.update(file_content)
        # 重置文件指针，以便后续读取
        file.seek(0)
    
    return hash_obj.hexdigest()


def validate_data_format(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    验证DataFrame格式是否符合要求。
    
    Args:
        df: 要验证的DataFrame
        
    Returns:
        (is_valid, error_messages): 是否有效和错误信息列表
    """
    errors = []
    
    # 检查必需列
    required_columns = ['article_title', 'journal_title', 'publication_year', 'abstract_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"缺少必需列: {missing_columns}")
    
    # 检查数据类型
    if 'publication_year' in df.columns:
        try:
            pd.to_numeric(df['publication_year'])
        except (ValueError, TypeError):
            errors.append("publication_year列包含无效的年份格式")
    
    # 检查数据质量
    if len(df) == 0:
        errors.append("数据为空")
    
    return len(errors) == 0, errors


class ProcessingError(Exception):
    """数据处理异常类"""
    pass


def get_processing_stats(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    """
    获取数据处理的统计信息。
    
    Args:
        original_df: 原始DataFrame（文件合并后的数据）
        processed_df: 处理后的DataFrame（最终处理完成的数据）
        
    Returns:
        包含统计信息的字典
    """
    # 计算各个处理步骤的统计信息
    stats = {
        'original_count': len(original_df),
        'processed_count': len(processed_df),
        'retention_rate': len(processed_df) / len(original_df) if len(original_df) > 0 else 0,
        'removed_total': len(original_df) - len(processed_df)
    }
    
    # 添加最终数据的基本统计（只有列存在时才计算）
    if len(processed_df) > 0:
        if 'publication_year' in processed_df.columns:
            stats['year_range'] = {
                'min': int(processed_df['publication_year'].min()),
                'max': int(processed_df['publication_year'].max())
            }
        
        if 'journal_title' in processed_df.columns:
            stats['journal_count'] = processed_df['journal_title'].nunique()
        
        if 'abstract_text' in processed_df.columns:
            stats['avg_abstract_length'] = processed_df['abstract_text'].str.len().mean()
    
    # 计算详细的处理步骤统计（只有原始数据存在时才计算）
    if len(original_df) > 0:
        # 检查必需列是否存在
        required_columns = ['article_title', 'abstract_text']
        missing_cols = [col for col in required_columns if col not in original_df.columns]
        
        if missing_cols:
            logger.warning(f"原始数据中缺少必需列: {missing_cols}")
            # 如果缺少必需列，无法计算详细统计
            return stats
        
        # 1. 去重统计
        try:
            dedup_count = original_df.drop_duplicates(
                subset=['article_title', 'abstract_text'], 
                keep='first'
            )
            stats['after_deduplication'] = len(dedup_count)
            stats['removed_by_deduplication'] = len(original_df) - len(dedup_count)
        except Exception as e:
            logger.warning(f"计算去重统计失败: {e}")
            stats['after_deduplication'] = len(original_df)
            stats['removed_by_deduplication'] = 0
        
        # 2. 缺失值处理统计
        try:
            # 检查是否有所有必需列
            missing_value_cols = ['article_title', 'journal_title', 'publication_year', 'abstract_text']
            available_cols = [col for col in missing_value_cols if col in dedup_count.columns]
            
            if available_cols:
                missing_value_count = dedup_count.dropna(subset=available_cols)
                stats['after_missing_values'] = len(missing_value_count)
                stats['removed_by_missing_values'] = len(dedup_count) - len(missing_value_count)
            else:
                stats['after_missing_values'] = len(dedup_count)
                stats['removed_by_missing_values'] = 0
        except Exception as e:
            logger.warning(f"计算缺失值统计失败: {e}")
            stats['after_missing_values'] = len(dedup_count) if 'dedup_count' in locals() else len(original_df)
            stats['removed_by_missing_values'] = 0
        
        # 3. 年份转换统计
        try:
            if 'missing_value_count' in locals() and 'publication_year' in missing_value_count.columns:
                year_converted = missing_value_count.copy()
                # 尝试转换为整数，失败的行会被移除
                year_mask = year_converted['publication_year'].apply(
                    lambda x: str(x).isdigit()
                )
                year_converted = year_converted[year_mask]
                year_converted['publication_year'] = year_converted['publication_year'].astype(int)
                stats['after_year_conversion'] = len(year_converted)
                stats['removed_by_year_conversion'] = len(missing_value_count) - len(year_converted)
            else:
                stats['after_year_conversion'] = stats.get('after_missing_values', len(original_df))
                stats['removed_by_year_conversion'] = 0
        except Exception as e:
            logger.warning(f"计算年份转换统计失败: {e}")
            stats['after_year_conversion'] = stats.get('after_missing_values', len(original_df))
            stats['removed_by_year_conversion'] = 0
        
        # 4. 内容过滤统计
        try:
            if 'year_converted' in locals() and 'abstract_text' in year_converted.columns:
                content_filtered = year_converted[year_converted['abstract_text'].str.len() >= 100]
                stats['after_content_filter'] = len(content_filtered)
                stats['removed_by_content_filter'] = len(year_converted) - len(content_filtered)
            else:
                stats['after_content_filter'] = stats.get('after_year_conversion', len(original_df))
                stats['removed_by_content_filter'] = 0
        except Exception as e:
            logger.warning(f"计算内容过滤统计失败: {e}")
            stats['after_content_filter'] = stats.get('after_year_conversion', len(original_df))
            stats['removed_by_content_filter'] = 0
        
        # 5. 文本预处理统计（使用最终处理结果）
        stats['after_text_preprocessing'] = len(processed_df)
        if 'content_filtered' in locals():
            stats['removed_by_text_preprocessing'] = len(content_filtered) - len(processed_df)
        else:
            stats['removed_by_text_preprocessing'] = 0
    
    return stats
