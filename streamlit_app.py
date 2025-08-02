# 文件名: streamlit_app.py

import streamlit as st
import zipfile
import io
from pathlib import Path
# 从我们即将创建的另外两个文件中，导入所有核心功能
from renamer import rename_files_in_memory
from processor import process_zip_in_memory

# --- 页面基础配置 ---
st.set_page_config(
    page_title="LOFTER 伴侣",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 LOFTER 伴侣")
st.caption("一个用于批量重命名和智能去除LOFTER图片水印的在线工具")

# --- 使用 Tab 来分隔两大功能 ---
tab1, tab2 = st.tabs(["1️⃣ 批量重命名工具", "2️⃣ 智能去水印工具"])


# ===================================================================
# ---                         功能一：批量重命名                       ---
# ===================================================================
with tab1:
    st.header("批量图片重命名")
    st.info("此工具用于将下载的LOFTER图片对，按时间顺序自动命名为 `[ID]-wm.jpg` 和 `[ID]-orig.jpg` 格式。")

    uploaded_files_rename = st.file_uploader(
        "上传包含图片对的文件夹 (或多选图片文件)",
        accept_multiple_files=True,
        key="renamer_uploader"
    )

    if uploaded_files_rename:
        st.success(f"已成功上传 {len(uploaded_files_rename)} 个文件。")
        if st.button("开始重命名", use_container_width=True):
            with st.spinner("正在分析并重命名图片..."):
                try:
                    # 调用内存中的重命名函数
                    renamed_zip_buffer, report = rename_files_in_memory(uploaded_files_rename)
                    
                    st.subheader("重命名报告:")
                    st.text(report) # 将详细报告打印出来
                    
                    st.download_button(
                        label="📥 下载已重命名的图片 (ZIP包)",
                        data=renamed_zip_buffer,
                        file_name="renamed_lofter_images.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"处理时发生错误: {e}")


# ===================================================================
# ---                         功能二：智能去水印                       ---
# ===================================================================
with tab2:
    st.header("智能AI去水印")
    st.info("此工具利用YOLOv8模型，自动识别并修复已按 `[ID]-wm.jpg` 和 `[ID]-orig.jpg` 格式命名的图片对。")
    st.warning("注意：AI模型加载和图片处理需要时间，请耐心等待。")
    
    with st.sidebar:
        st.header("AI去水印配置")
        conf_threshold = st.slider("模型自信度门槛", 0.1, 1.0, 0.5, 0.05)
        
        st.subheader("搜索区域 (比例)")
        col1, col2 = st.columns(2)
        search_x_start = col1.slider("左", 0.0, 1.0, 0.0, 0.05)
        search_x_end = col2.slider("右", 0.0, 1.0, 1.0, 0.05)
        search_y_start = col1.slider("上", 0.0, 1.0, 0.5, 0.05)
        search_y_end = col2.slider("下", 0.0, 1.0, 1.0, 0.05)
        
        st.subheader("修复区域扩大比例")
        width_exp = st.slider("宽度扩大", 0.0, 1.0, 0.2, 0.05)
        height_exp = st.slider("高度扩大", 0.0, 1.0, 0.1, 0.05)
        
        # 将配置打包成一个字典
        processor_config = {
            'YOLO_CONFIDENCE_THRESHOLD': conf_threshold,
            'SEARCH_REGION_RATIOS': (search_x_start, search_y_start, search_x_end, search_y_end),
            'WIDTH_EXPANSION_RATIO': width_exp,
            'HEIGHT_EXPANSION_RATIO': height_exp
        }

    uploaded_zip_process = st.file_uploader(
        "上传包含已重命名图片对的ZIP包",
        type="zip",
        key="processor_uploader"
    )

    if uploaded_zip_process:
        st.success("ZIP文件上传成功！")
        if st.button("开始AI去水印", use_container_width=True):
            status_area = st.empty()
            with st.spinner("AI正在全力工作中，这可能需要几分钟..."):
                try:
                    # 调用内存中的处理函数
                    repaired_zip_buffer, report = process_zip_in_memory(
                        uploaded_zip_process, 
                        processor_config, 
                        status_area
                    )

                    status_area.empty()
                    st.subheader("处理报告:")
                    st.text(report) # 将详细报告打印出来
                    
                    st.download_button(
                        label="📥 下载已修复的图片 (ZIP包)",
                        data=repaired_zip_buffer,
                        file_name="repaired_lofter_images.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                except Exception as e:
                    status_area.empty()
                    st.error(f"处理时发生错误: {e}")
                    st.code(str(e)) # 打印详细错误以供调试