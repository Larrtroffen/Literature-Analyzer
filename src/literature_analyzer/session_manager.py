import json
import pickle
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import os
import base64

class SessionManager:
    def __init__(self, session_dir: str = "sessions"):
        self.session_dir = session_dir
        self.ensure_session_dir()

    def ensure_session_dir(self):
        """确保会话目录存在"""
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

    def save_session(self, session_data: Dict[str, Any], session_name: str = None) -> str:
        """保存当前会话状态"""
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 准备要保存的数据
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'session_name': session_name,
            'data': session_data
        }
        
        # 保存为JSON文件（对于可序列化的数据）
        json_file = os.path.join(self.session_dir, f"{session_name}.json")
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=self._json_serializer)
        except (TypeError, ValueError) as e:
            st.warning(f"无法保存为JSON格式: {e}")
        
        # 保存为pickle文件（对于复杂对象）
        pickle_file = os.path.join(self.session_dir, f"{session_name}.pkl")
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(save_data, f)
        except (pickle.PicklingError, EOFError) as e:
            st.error(f"保存会话失败: {e}")
            return None
        
        return session_name

    def load_session(self, session_name: str) -> Optional[Dict[str, Any]]:
        """加载会话状态"""
        # 首先尝试加载pickle文件
        pickle_file = os.path.join(self.session_dir, f"{session_name}.pkl")
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    session_data = pickle.load(f)
                return session_data.get('data', {})
            except (pickle.UnpicklingError, EOFError) as e:
                st.warning(f"无法从pickle文件加载会话: {e}")
        
        # 如果pickle文件失败，尝试加载JSON文件
        json_file = os.path.join(self.session_dir, f"{session_name}.json")
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                return session_data.get('data', {})
            except (json.JSONDecodeError, FileNotFoundError) as e:
                st.error(f"无法从JSON文件加载会话: {e}")
        
        return None

    def list_sessions(self) -> pd.DataFrame:
        """列出所有可用的会话"""
        sessions = []
        
        if os.path.exists(self.session_dir):
            for filename in os.listdir(self.session_dir):
                if filename.endswith('.json'):
                    session_name = filename[:-5]  # 移除.json扩展名
                    json_file = os.path.join(self.session_dir, filename)
                    
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        sessions.append({
                            'session_name': session_name,
                            'timestamp': session_data.get('timestamp', ''),
                            'file_size': os.path.getsize(json_file)
                        })
                    except (json.JSONDecodeError, FileNotFoundError):
                        continue
        
        if sessions:
            return pd.DataFrame(sessions).sort_values('timestamp', ascending=False)
        else:
            return pd.DataFrame(columns=['session_name', 'timestamp', 'file_size'])

    def delete_session(self, session_name: str) -> bool:
        """删除指定的会话"""
        deleted = False
        
        # 删除JSON文件
        json_file = os.path.join(self.session_dir, f"{session_name}.json")
        if os.path.exists(json_file):
            try:
                os.remove(json_file)
                deleted = True
            except OSError as e:
                st.error(f"删除JSON文件失败: {e}")
        
        # 删除pickle文件
        pickle_file = os.path.join(self.session_dir, f"{session_name}.pkl")
        if os.path.exists(pickle_file):
            try:
                os.remove(pickle_file)
                deleted = True
            except OSError as e:
                st.error(f"删除pickle文件失败: {e}")
        
        return deleted

    def export_session(self, session_name: str) -> Optional[str]:
        """导出会话为可下载的文件"""
        pickle_file = os.path.join(self.session_dir, f"{session_name}.pkl")
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    data = f.read()
                
                # 创建下载链接
                b64 = base64.b64encode(data).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{session_name}.pkl">下载会话文件</a>'
                return href
            except Exception as e:
                st.error(f"导出会话失败: {e}")
        
        return None

    def import_session(self, uploaded_file) -> Optional[str]:
        """从上传的文件导入会话"""
        try:
            session_data = pickle.load(uploaded_file)
            session_name = session_data.get('session_name', f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # 保存导入的会话
            return self.save_session(session_data.get('data', {}), session_name)
        except Exception as e:
            st.error(f"导入会话失败: {e}")
            return None

    def _json_serializer(self, obj):
        """JSON序列化器，处理特殊对象类型"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    @staticmethod
    def get_current_session_state() -> Dict[str, Any]:
        """获取当前Streamlit会话状态中需要保存的部分"""
        # 定义需要保存的键
        save_keys = [
            'df_processed', 'embeddings', 'reduced_embeddings', 'topic_model',
            'topic_info', 'df_with_topics', 'selected_embedding_model',
            'selected_topic_model', 'selected_topic_model_type',
            'global_filters', 'custom_topic_names'
        ]
        
        session_data = {}
        for key in save_keys:
            if key in st.session_state:
                session_data[key] = st.session_state[key]
        
        return session_data

    @staticmethod
    def restore_session_state(session_data: Dict[str, Any]):
        """恢复会话状态到Streamlit"""
        for key, value in session_data.items():
            st.session_state[key] = value

    def create_session_ui(self):
        """创建会话管理的UI界面"""
        st.subheader("会话管理")
        
        # 获取会话列表
        sessions_df = self.list_sessions()
        
        # 创建两列布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("### 保存当前会话")
            session_name = st.text_input("会话名称", value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if st.button("保存会话"):
                current_state = self.get_current_session_state()
                if current_state:
                    saved_name = self.save_session(current_state, session_name)
                    if saved_name:
                        st.success(f"会话 '{saved_name}' 已保存")
                        st.rerun()
                else:
                    st.warning("没有可保存的会话数据")
        
        with col2:
            st.write("### 加载会话")
            if not sessions_df.empty:
                selected_session = st.selectbox(
                    "选择要加载的会话",
                    sessions_df['session_name'].tolist(),
                    format_func=lambda x: f"{x} ({sessions_df[sessions_df['session_name'] == x]['timestamp'].iloc[0]})"
                )
                
                if st.button("加载会话"):
                    loaded_data = self.load_session(selected_session)
                    if loaded_data:
                        self.restore_session_state(loaded_data)
                        st.success(f"会话 '{selected_session}' 已加载")
                        st.rerun()
                    else:
                        st.error("加载会话失败")
                
                # 删除和导出按钮
                col_del, col_exp = st.columns([1, 1])
                
                with col_del:
                    if st.button("删除会话"):
                        if self.delete_session(selected_session):
                            st.success(f"会话 '{selected_session}' 已删除")
                            st.rerun()
                
                with col_exp:
                    export_link = self.export_session(selected_session)
                    if export_link:
                        st.markdown(export_link, unsafe_allow_html=True)
            else:
                st.info("没有可用的会话")
        
        # 导入会话
        st.write("### 导入会话")
        uploaded_file = st.file_uploader("上传会话文件 (.pkl)", type=['pkl'])
        if uploaded_file is not None:
            if st.button("导入会话"):
                imported_name = self.import_session(uploaded_file)
                if imported_name:
                    st.success(f"会话 '{imported_name}' 已导入")
                    st.rerun()
