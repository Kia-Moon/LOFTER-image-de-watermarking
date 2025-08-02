# 文件名: processor.py

import io
import zipfile
import re
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import streamlit as st

# --- 缓存模型加载 ---
# 使用Streamlit的缓存功能，让模型只在第一次加载，极大提高后续运行速度
@st.cache_resource
def load_yolo_model():
    # 注意：这里的路径是相对于项目根目录的
    # 你需要把 best.pt 文件也上传到你的GitHub仓库
    model_path = "best.pt" 
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise FileNotFoundError(f"无法加载YOLO模型！请确保 'best.pt' 文件在项目主目录中。错误: {e}")

def repair_image_in_memory(wm_data, orig_data, model, config):
    """在内存中对单对图片进行修复"""
    # 将字节数据解码为OpenCV图像
    wm_img_np = np.frombuffer(wm_data, np.uint8)
    orig_img_np = np.frombuffer(orig_data, np.uint8)
    high_res_img = cv2.imdecode(wm_img_np, cv2.IMREAD_COLOR)
    low_res_img = cv2.imdecode(orig_img_np, cv2.IMREAD_COLOR)
    
    if high_res_img is None or low_res_img is None: 
        return None, "图片解码失败"

    h_high, w_high, _ = high_res_img.shape
    
    search_x_start = int(w_high * config['SEARCH_REGION_RATIOS'][0])
    search_y_start = int(h_high * config['SEARCH_REGION_RATIOS'][1])
    search_x_end = int(w_high * config['SEARCH_REGION_RATIOS'][2])
    search_y_end = int(h_high * config['SEARCH_REGION_RATIOS'][3])
    
    search_region = high_res_img[search_y_start:search_y_end, search_x_start:search_x_end]
    
    results = model.predict(source=search_region, conf=config['YOLO_CONFIDENCE_THRESHOLD'], verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, "未在指定区域内定位到水印"

    all_xyxy = boxes.xyxy.cpu().numpy()
    x_min_rel = int(np.min(all_xyxy[:, 0]))
    y_min_rel = int(np.min(all_xyxy[:, 1]))
    x_max_rel = int(np.max(all_xyxy[:, 2]))
    y_max_rel = int(np.max(all_xyxy[:, 3]))

    x_min_abs = x_min_rel + search_x_start
    y_min_abs = y_min_rel + search_y_start
    x_max_abs = x_max_rel + search_x_start
    y_max_abs = y_max_rel + search_y_start
    
    original_width = x_max_abs - x_min_abs
    original_height = y_max_abs - y_min_abs
    
    width_margin = int((original_width * config['WIDTH_EXPANSION_RATIO']) / 2)
    height_margin = int((original_height * config['HEIGHT_EXPANSION_RATIO']) / 2)
    
    x_start = max(0, x_min_abs - width_margin)
    y_start = max(0, y_min_abs - height_margin)
    x_end = min(w_high, x_max_abs + width_margin)
    y_end = min(h_high, y_max_abs + height_margin)
    
    low_res_resized = cv2.resize(low_res_img, (w_high, h_high), interpolation=cv2.INTER_LANCZOS4)
    clean_patch = low_res_resized[y_start:y_end, x_start:x_end]

    if clean_patch.shape[0] == 0 or clean_patch.shape[1] == 0:
        return None, "修复补丁计算尺寸无效"
    
    high_res_img[y_start:y_end, x_start:x_end] = clean_patch
    
    # 将修复后的图像编码回字节流
    _, buffer = cv2.imencode('.jpg', high_res_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return buffer.tobytes(), "修复成功"

def process_zip_in_memory(zip_file_obj, config, status_area):
    """在内存中处理上传的ZIP包"""
    model = load_yolo_model()
    report_lines = ["--- AI去水印处理报告 ---"]
    
    input_zip = zipfile.ZipFile(zip_file_obj, 'r')
    files_map = {Path(f).stem: f for f in input_zip.namelist()}
    
    wm_pattern = re.compile(r"(.+)-wm$")
    tasks = []
    
    for base_name_stem, full_path in files_map.items():
        m = wm_pattern.match(base_name_stem)
        if m:
            base_id = m.group(1)
            orig_full_path = files_map.get(f"{base_id}-orig")
            if orig_full_path:
                tasks.append((full_path, orig_full_path))
    
    if not tasks:
        raise ValueError("ZIP包中未找到任何有效的图片对 (如 'id-wm.jpg' 和 'id-orig.jpg')")

    output_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(output_zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as output_zip:
        for i, (wm_path, orig_path) in enumerate(tasks):
            status_area.text(f"正在处理第 {i+1}/{len(tasks)} 对图片: {Path(wm_path).name}")
            
            wm_data = input_zip.read(wm_path)
            orig_data = input_zip.read(orig_path)
            
            repaired_data, message = repair_image_in_memory(wm_data, orig_data, model, config)
            
            if repaired_data:
                # 修复后的文件名保持和有水印图一致
                output_zip.writestr(Path(wm_path).name, repaired_data)
                report_lines.append(f"  [成功] {Path(wm_path).name} - {message}")
            else:
                report_lines.append(f"  [失败] {Path(wm_path).name} - {message}")

    report = "\n".join(report_lines)
    output_zip_buffer.seek(0)
    return output_zip_buffer, report