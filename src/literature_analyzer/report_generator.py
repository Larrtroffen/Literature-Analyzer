import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any, Optional

class ReportGenerator:
    def __init__(self):
        self.report_content = []
        self.title = "文献分析报告"
        self.introduction = ""
        self.conclusion = ""

    def set_metadata(self, title: str, introduction: str = "", conclusion: str = ""):
        """设置报告的元数据"""
        self.title = title
        self.introduction = introduction
        self.conclusion = conclusion

    def add_section(self, title: str, content: str):
        """添加一个文本章节"""
        self.report_content.append({
            'type': 'text',
            'title': title,
            'content': content
        })

    def add_table(self, title: str, df: pd.DataFrame, max_rows: int = 20):
        """添加一个数据表格"""
        if len(df) > max_rows:
            df_display = df.head(max_rows)
            note = f"\n\n*注：表格仅显示前 {max_rows} 行，共 {len(df)} 行数据。*"
        else:
            df_display = df
            note = ""
        
        # 转换为HTML表格
        html_table = df_display.to_html(classes='table table-striped', index=False)
        
        self.report_content.append({
            'type': 'table',
            'title': title,
            'content': html_table + note
        })

    def add_chart(self, title: str, fig: go.Figure, description: str = ""):
        """添加一个图表"""
        # 将图表转换为HTML
        chart_html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{len(self.report_content)}")
        
        self.report_content.append({
            'type': 'chart',
            'title': title,
            'content': chart_html,
            'description': description
        })

    def add_statistics(self, title: str, stats: Dict[str, Any]):
        """添加统计信息"""
        stats_html = "<ul>"
        for key, value in stats.items():
            stats_html += f"<li><strong>{key}:</strong> {value}</li>"
        stats_html += "</ul>"
        
        self.report_content.append({
            'type': 'statistics',
            'title': title,
            'content': stats_html
        })

    def generate_html_report(self) -> str:
        """生成HTML格式的报告"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.6; }}
        .header {{ background-color: #f8f9fa; padding: 2rem 0; margin-bottom: 2rem; }}
        .section {{ margin-bottom: 3rem; }}
        .section-title {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }}
        .table {{ margin-top: 1rem; }}
        .stats-box {{ background-color: #e8f4f8; padding: 1rem; border-radius: 5px; margin: 1rem 0; }}
        .chart-container {{ margin: 2rem 0; }}
        .footer {{ background-color: #f8f9fa; padding: 1rem 0; margin-top: 3rem; text-align: center; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1>{self.title}</h1>
            <p class="lead">生成时间: {current_time}</p>
        </div>

        {self._generate_introduction_html()}

        {self._generate_content_html()}

        {self._generate_conclusion_html()}

        <div class="footer">
            <p>本报告由文献分析应用自动生成</p>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        """
        
        return html_template

    def _generate_introduction_html(self) -> str:
        """生成引言部分的HTML"""
        if not self.introduction:
            return ""
        
        return f"""
        <div class="section">
            <h2 class="section-title">引言</h2>
            <p>{self.introduction}</p>
        </div>
        """

    def _generate_conclusion_html(self) -> str:
        """生成结论部分的HTML"""
        if not self.conclusion:
            return ""
        
        return f"""
        <div class="section">
            <h2 class="section-title">结论</h2>
            <p>{self.conclusion}</p>
        </div>
        """

    def _generate_content_html(self) -> str:
        """生成主要内容部分的HTML"""
        content_html = ""
        
        for section in self.report_content:
            section_html = f"""
            <div class="section">
                <h2 class="section-title">{section['title']}</h2>
            """
            
            if section['type'] == 'text':
                section_html += f"<p>{section['content']}</p>"
            elif section['type'] == 'table':
                section_html += section['content']
            elif section['type'] == 'chart':
                section_html += f"<p>{section['description']}</p>"
                section_html += f'<div class="chart-container">{section["content"]}</div>'
            elif section['type'] == 'statistics':
                section_html += f'<div class="stats-box">{section["content"]}</div>'
            
            section_html += "</div>"
            content_html += section_html
        
        return content_html

    def generate_pdf_report(self) -> bytes:
        """生成PDF格式的报告（需要安装weasyprint）"""
        try:
            from weasyprint import HTML, CSS
            
            html_content = self.generate_html_report()
            
            # 添加PDF特定的CSS样式
            pdf_css = CSS(string="""
                @page { size: A4; margin: 2cm; }
                body { font-size: 11pt; }
                .header { page-break-after: avoid; }
                .section { page-break-inside: avoid; }
                .chart-container { page-break-inside: avoid; }
            """)
            
            # 生成PDF
            pdf_buffer = BytesIO()
            HTML(string=html_content).write_pdf(pdf_buffer, stylesheets=[pdf_css])
            pdf_buffer.seek(0)
            
            return pdf_buffer.read()
            
        except ImportError:
            st.error("生成PDF需要安装weasyprint。请运行: pip install weasyprint")
            return b""

    def download_html(self) -> str:
        """获取HTML报告的下载链接"""
        html_content = self.generate_html_report()
        b64 = base64.b64encode(html_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="literature_analysis_report.html">下载HTML报告</a>'
        return href

    def download_pdf(self) -> Optional[str]:
        """获取PDF报告的下载链接"""
        pdf_content = self.generate_pdf_report()
        if pdf_content:
            b64 = base64.b64encode(pdf_content).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="literature_analysis_report.pdf">下载PDF报告</a>'
            return href
        return None

    @staticmethod
    def create_sample_report(df: pd.DataFrame, topic_info: pd.DataFrame, temporal_data: pd.DataFrame = None) -> 'ReportGenerator':
        """创建示例报告的工厂方法"""
        report = ReportGenerator()
        
        # 设置报告元数据
        report.set_metadata(
            title="文献分析报告",
            introduction="本报告基于导入的文献数据，通过主题建模、时间趋势分析等方法，对文献集进行了全面的分析。",
            conclusion="通过本次分析，我们识别出了主要的研究主题和发展趋势，为后续研究提供了有价值的参考。"
        )
        
        # 添加数据概览
        report.add_statistics(
            "数据概览",
            {
                "文献总数": len(df),
                "期刊数量": df['journal_title'].nunique(),
                "发表年份范围": f"{df['publication_year'].min()} - {df['publication_year'].max()}",
                "主题数量": len(topic_info) if not topic_info.empty else 0
            }
        )
        
        # 添加主题信息表格
        if not topic_info.empty:
            report.add_table("主题分布", topic_info[['Topic', 'Name', 'Count']])
        
        # 添加时间趋势分析
        if temporal_data is not None and not temporal_data.empty:
            yearly_counts = df['publication_year'].value_counts().sort_index()
            if not yearly_counts.empty:
                fig = px.line(x=yearly_counts.index, y=yearly_counts.values, 
                             title="年度发表趋势", labels={'x': '年份', 'y': '文献数量'})
                report.add_chart("年度发表趋势", fig, "展示了文献发表数量的年度变化趋势。")
        
        return report
