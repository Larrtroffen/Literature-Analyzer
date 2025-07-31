import openai
import streamlit as st
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMProvider(ABC):
    """LLM提供商抽象基类"""
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """生成文本的抽象方法"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查提供商是否可用"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI提供商实现"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key if api_key else st.secrets.get("OPENAI_API_KEY")
        self.base_url = base_url if base_url else st.secrets.get("OPENAI_BASE_URL")
        self.model_name = model_name
        self.client = None
        
        if self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                st.error(f"初始化OpenAI客户端失败: {e}")
        else:
            st.warning("OpenAI API Key 未配置")
    
    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        if not self.client:
            return "OpenAI客户端未初始化"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            st.error(f"OpenAI API 调用失败: {e.status_code} - {e.response}")
            return "OpenAI API 调用失败"
        except Exception as e:
            st.error(f"OpenAI 调用失败: {e}")
            return "OpenAI 调用失败"
    
    def is_available(self) -> bool:
        return self.client is not None


class LocalLLMProvider(LLMProvider):
    """本地LLM提供商实现"""
    
    def __init__(self, model_path: str = None, api_base: str = "http://localhost:8000"):
        self.model_path = model_path
        self.api_base = api_base
        self.is_configured = model_path is not None
        
        if not self.is_configured:
            st.warning("本地LLM模型路径未配置")
    
    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        if not self.is_configured:
            return "本地LLM未配置"
        
        try:
            # 这里可以集成本地LLM的调用逻辑
            # 例如使用requests调用本地API
            import requests
            
            response = requests.post(
                f"{self.api_base}/generate",
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("text", "本地LLM响应为空")
            else:
                return f"本地LLM调用失败: {response.status_code}"
                
        except Exception as e:
            st.error(f"本地LLM调用失败: {e}")
            return "本地LLM调用失败"
    
    def is_available(self) -> bool:
        return self.is_configured


class LLMClient:
    """LLM客户端，支持多种提供商"""
    
    def __init__(self, provider: LLMProvider = None):
        if provider is None:
            # 默认尝试使用OpenAI
            self.provider = OpenAIProvider()
        else:
            self.provider = provider
        
        if not self.provider.is_available():
            st.warning("LLM提供商不可用，部分AI功能可能受限")
    
    @st.cache_data(show_spinner=False)
    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """生成文本"""
        return self.provider.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
    
    def is_available(self) -> bool:
        """检查LLM是否可用"""
        return self.provider.is_available()
