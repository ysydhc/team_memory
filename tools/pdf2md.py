#!/usr/bin/env python3
"""
pdf2md.py — PDF 转 Markdown 增强工具

用法:
    python pdf2md.py input.pdf                    # 输出到 stdout
    python pdf2md.py input.pdf -o output.md       # 输出到文件
    python pdf2md.py input.pdf --pages 1-5        # 只转换前 5 页
    python pdf2md.py input.pdf --images-dir ./img  # 提取图片到指定目录

特性:
    - 标题识别: 根据字号 + 字体特征自动推断 h1~h4
    - 列表识别: 检测 •/●/◆/➤/-/数字. 等列表标记
    - 表格识别: 等宽多列对齐文本 → Markdown 表格
    - 图片提取: 导出为 PNG 并插入 ![](path) 引用
    - 页面分隔: 可选 --- 分页线
"""

import argparse
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# pymupdf 安装位置兼容
try:
    import pymupdf
except ImportError:
    sys.path.insert(0, "/tmp/pymupdf_test")
    import pymupdf


# ---------------------------------------------------------------------------
# 1. 标题检测
# ---------------------------------------------------------------------------

def detect_heading_level(span, font_size_stats):
    """
    根据字号和字体特征判断标题层级。
    
    策略:
    - 字号 > body_max * 1.8        → h1
    - 字号 > body_max * 1.4        → h2
    - 字号 > body_max * 1.15       → h3
    - 字号 > body_max * 1.05       → h4
    - 粗体 + 稍大字号              → 提升一级
    """
    size = span["size"]
    font = span.get("font", "").lower()
    flags = span.get("flags", 0)
    is_bold = bool(flags & 2 ** 4) or "bold" in font or "black" in font
    
    body_max = font_size_stats.get("body_max", 12)
    
    ratio = size / body_max if body_max > 0 else 1
    
    if ratio > 1.6:
        level = 1
    elif ratio > 1.25:
        level = 2
    elif ratio > 1.1:
        level = 3
    elif ratio > 1.0:
        level = 4
    else:
        return 0  # 不是标题
    
    # 粗体提升一级
    if is_bold and level < 4:
        level -= 1
    
    return level


def compute_font_stats(page_dicts):
    """统计全文档字号分布，找出正文字号。"""
    size_count = defaultdict(int)
    
    for page_data in page_dicts:
        for block in page_data.get("blocks", []):
            if block["type"] != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size_count[span["size"]] += 1
    
    if not size_count:
        return {"body_max": 12, "body_mode": 12}
    
    # 出现最多的字号视为正文（众数）
    body_mode = max(size_count, key=size_count.get)
    # 用众数作为 body 基准，而非最大常见字号
    # 因为最大常见字号可能包含标题字号
    body_max = body_mode
    
    return {"body_max": body_max, "body_mode": body_mode}


# ---------------------------------------------------------------------------
# 2. 列表检测
# ---------------------------------------------------------------------------

LIST_BULLET_RE = re.compile(r'^[\s]*([•●◆➤▪►\-–—·\*])\s+')
LIST_NUMBER_RE = re.compile(r'^[\s]*(\d{1,2}[\.\)、])\s+(?!\d)')  # 避免匹配 1.1 这种章节号
LIST_ALPHA_RE  = re.compile(r'^[\s]*([a-zA-Z][\.\)])\s+')

def detect_list_item(text):
    """
    检测列表项，返回 (indent_level, marker, content) 或 None。
    indent_level: 0=顶级, 1=缩进一级, 2+=更深
    """
    m = LIST_NUMBER_RE.match(text)
    if m:
        indent = _indent_level(text, m.start(1))
        return (indent, m.group(1), text[m.end():].strip())
    
    m = LIST_ALPHA_RE.match(text)
    if m:
        indent = _indent_level(text, m.start(1))
        return (indent, m.group(1), text[m.end():].strip())
    
    m = LIST_BULLET_RE.match(text)
    if m:
        indent = _indent_level(text, m.start(1))
        return (indent, m.group(1), text[m.end():].strip())
    
    return None


def _indent_level(text, marker_pos):
    """根据 marker 前的空格数推断缩进层级。"""
    prefix = text[:marker_pos]
    spaces = len(prefix) - len(prefix.lstrip())
    if spaces < 10:
        return 0
    elif spaces < 25:
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# 3. 表格检测
# ---------------------------------------------------------------------------

def is_table_block(block):
    """
    判断一个 block 是否可能是表格。
    
    线索:
    - 多行文本，行内包含多个连续空格（列分隔符）
    - 存在制表符或对齐的列
    """
    if block["type"] != 0:
        return False
    
    lines = block.get("lines", [])
    if len(lines) < 2:
        return False
    
    multi_space_lines = 0
    separator_line = False
    
    for line in lines:
        text = ""
        for span in line.get("spans", []):
            text += span.get("text", "")
        
        # 多个连续空格 = 列分隔
        if re.search(r'\s{3,}', text.strip()):
            multi_space_lines += 1
        
        # 分隔线 (─ ━ ─ ┍ 等)
        if re.match(r'^[\s\-─━═╍┄┈]+$', text.strip()):
            separator_line = True
    
    # 至少 2 行有多列特征，或有分隔线
    return multi_space_lines >= 2 or (multi_space_lines >= 1 and separator_line)


def block_to_table(block):
    """
    将疑似表格的 block 转为 Markdown 表格。
    
    策略: 基于每行中各 span 的 x 坐标对齐来分列。
    """
    lines = block.get("lines", [])
    if not lines:
        return None
    
    # 收集所有列的 x 坐标
    col_positions = []
    rows = []
    
    for line in lines:
        line_text = ""
        for span in line.get("spans", []):
            line_text += span.get("text", "")
        
        # 跳过分隔线
        stripped = line_text.strip()
        if re.match(r'^[\-\─\━\═\╍\┄\┈\·]+$', stripped):
            continue
        
        # 按连续空格分列
        cols = re.split(r'\s{2,}', stripped)
        cols = [c.strip() for c in cols if c.strip()]
        
        if cols:
            rows.append(cols)
            if len(cols) > len(col_positions):
                col_positions = list(range(len(cols)))
    
    if len(rows) < 2:
        return None
    
    # 标准化列数
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")
    
    # 构建 Markdown 表格
    md_lines = []
    
    # 表头
    header = rows[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    # 数据行
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# 4. 图片提取
# ---------------------------------------------------------------------------

def extract_images(page, page_num, images_dir):
    """提取页面中的图片，返回 [(image_path, bbox), ...]"""
    if not images_dir:
        return []
    
    results = []
    image_list = page.get_images(full=True)
    
    for img_idx, img_info in enumerate(image_list):
        xref = img_info[0]
        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            
            filename = f"page{page_num + 1}_img{img_idx + 1}.{image_ext}"
            filepath = os.path.join(images_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            results.append((filepath, image_ext))
        except Exception:
            continue
    
    return results


# ---------------------------------------------------------------------------
# 5. 主转换逻辑
# ---------------------------------------------------------------------------

def convert_pdf_to_markdown(pdf_path, pages=None, output_path=None,
                            images_dir=None, page_separator=True):
    """
    将 PDF 文件转为 Markdown。
    
    Args:
        pdf_path: PDF 文件路径
        pages: 页码范围，如 "1-5" 或 "3,5,7"
        output_path: 输出文件路径，None 则输出到 stdout
        images_dir: 图片提取目录，None 则不提取
        page_separator: 是否在页面间插入分隔线
    
    Returns:
        Markdown 文本
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    
    doc = pymupdf.open(str(pdf_path))
    
    # 解析页码范围
    page_indices = _parse_page_range(pages, len(doc))
    
    # 第一遍: 收集所有页面结构，计算字号统计
    page_dicts = []
    for idx in page_indices:
        page_dicts.append(doc[idx].get_text("dict"))
    
    font_stats = compute_font_stats(page_dicts)
    
    # 第二遍: 逐页转换
    md_parts = []
    
    for page_idx, page_data in zip(page_indices, page_dicts):
        page = doc[page_idx]
        
        # 提取图片
        img_refs = {}
        if images_dir:
            os.makedirs(images_dir, exist_ok=True)
            extracted = extract_images(page, page_idx, images_dir)
            for img_path, img_ext in extracted:
                img_refs[os.path.basename(img_path)] = img_path
        
        page_md = _convert_page(page_data, font_stats, page_idx, img_refs)
        md_parts.append(page_md)
    
    doc.close()
    
    # 组装最终输出
    separator = "\n\n---\n\n" if page_separator else "\n\n"
    result = separator.join(md_parts)
    
    # 清理多余空行
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    result = result.strip() + "\n"
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Written to {output_path}", file=sys.stderr)
    
    return result


def _convert_page(page_data, font_stats, page_num, img_refs):
    """转换单个页面。"""
    blocks = page_data.get("blocks", [])
    md_lines = []
    
    # 用于跟踪上一个 block 类型，决定空行
    prev_type = None
    
    for block in blocks:
        # --- 图片 block ---
        if block["type"] == 1:
            # pymupdf 图片 block
            width = block.get("width", 0)
            height = block.get("height", 0)
            bbox = block.get("bbox", [])
            
            # 尝试匹配已提取的图片
            img_md = None
            for basename, fpath in img_refs.items():
                if f"page{page_num + 1}" in basename:
                    img_md = f"![{basename}]({fpath})"
                    break
            
            if img_md is None:
                img_md = f"![image page {page_num + 1}]"
            
            _add_with_spacing(md_lines, img_md, prev_type, "image")
            prev_type = "image"
            continue
        
        # --- 文本 block ---
        if block["type"] != 0:
            continue
        
        # 检查是否为表格
        if is_table_block(block):
            table_md = block_to_table(block)
            if table_md:
                _add_with_spacing(md_lines, table_md, prev_type, "table")
                prev_type = "table"
                continue
        
        # 普通文本 block: 逐行处理
        lines = block.get("lines", [])
        block_text_parts = []
        
        for line in lines:
            line_text = ""
            dominant_span = None
            max_len = 0
            
            for span in line.get("spans", []):
                text = span.get("text", "")
                line_text += text
                if len(text) > max_len:
                    max_len = len(text)
                    dominant_span = span
            
            if not line_text.strip():
                continue
            
            # 检测标题
            if dominant_span:
                heading = detect_heading_level(dominant_span, font_stats)
                if heading > 0:
                    # 如果之前有未输出的正文，先输出
                    if block_text_parts:
                        combined = " ".join(block_text_parts)
                        _add_with_spacing(md_lines, combined, prev_type, "text")
                        block_text_parts = []
                    
                    heading_md = f"{'#' * heading} {line_text.strip()}"
                    _add_with_spacing(md_lines, heading_md, prev_type, "heading")
                    prev_type = "heading"
                    continue
            
            # 检测列表项
            list_item = detect_list_item(line_text)
            if list_item:
                indent, marker, content = list_item
                # 如果之前有未输出的正文，先输出
                if block_text_parts:
                    combined = " ".join(block_text_parts)
                    _add_with_spacing(md_lines, combined, prev_type, "text")
                    block_text_parts = []
                
                # 把行内的破折号/中间点还原为 — 
                content = re.sub(r'\s+[·]\s+', ' — ', content)
                
                indent_str = "  " * indent
                # 数字列表 vs 无序列表
                if re.match(r'\d+', marker):
                    list_md = f"{indent_str}{marker} {content}"
                else:
                    list_md = f"{indent_str}- {content}"
                
                _add_with_spacing(md_lines, list_md, prev_type, "list")
                prev_type = "list"
                continue
            
            # 检测列表续行：前一个是列表项，当前行短且以空格开头（缩进续行）
            if prev_type == "list" and line_text.startswith("  ") and len(line_text.strip()) < 80:
                # 追加到上一个列表项
                if md_lines:
                    md_lines[-1] = md_lines[-1].rstrip() + " " + line_text.strip()
                prev_type = "list"
                continue
            
            # 检测图片说明 (Figure / Fig / 图)
            if re.match(r'^(Figure|Fig\.?|图)\s*\d', line_text.strip(), re.IGNORECASE):
                if block_text_parts:
                    combined = " ".join(block_text_parts)
                    _add_with_spacing(md_lines, combined, prev_type, "text")
                    block_text_parts = []
                _add_with_spacing(md_lines, f"*{line_text.strip()}*", prev_type, "caption")
                prev_type = "caption"
                continue
            
            # 普通正文
            block_text_parts.append(line_text.strip())
        
        # 输出剩余正文
        if block_text_parts:
            combined = " ".join(block_text_parts)
            # 合并同一 block 的断行
            combined = re.sub(r'\s{2,}', ' ', combined)
            _add_with_spacing(md_lines, combined, prev_type, "text")
            prev_type = "text"
    
    return "\n".join(md_lines)


def _add_with_spacing(md_lines, new_content, prev_type, curr_type):
    """根据上下文类型添加适当的空行分隔。"""
    # 标题前后都需要空行
    if curr_type == "heading" or prev_type == "heading":
        if md_lines and md_lines[-1] != "":
            md_lines.append("")
    # 表格前后空行
    elif curr_type == "table" or prev_type == "table":
        if md_lines and md_lines[-1] != "":
            md_lines.append("")
    # 列表项之间不加空行，但列表与其他类型之间要加
    elif curr_type == "list" and prev_type and prev_type != "list":
        if md_lines and md_lines[-1] != "":
            md_lines.append("")
    elif prev_type == "list" and curr_type != "list":
        if md_lines and md_lines[-1] != "":
            md_lines.append("")
    
    md_lines.append(new_content)


def _parse_page_range(pages_str, total_pages):
    """解析页码范围字符串，返回 0-based 索引列表。"""
    if not pages_str:
        return list(range(total_pages))
    
    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start) - 1  # 转为 0-based
            end = int(end)          # end 是 inclusive，直接用
            indices.update(range(max(0, start), min(end, total_pages)))
        else:
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                indices.add(idx)
    
    return sorted(indices)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PDF → Markdown 增强转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input", help="PDF 文件路径")
    parser.add_argument("-o", "--output", help="输出 Markdown 文件路径")
    parser.add_argument("--pages", help="页码范围，如 '1-5' 或 '3,5,7'")
    parser.add_argument("--images-dir", help="图片提取目录")
    parser.add_argument("--no-page-sep", action="store_true",
                        help="不在页面间插入分隔线")
    
    args = parser.parse_args()
    
    md = convert_pdf_to_markdown(
        pdf_path=args.input,
        pages=args.pages,
        output_path=args.output,
        images_dir=args.images_dir,
        page_separator=not args.no_page_sep,
    )
    
    if not args.output:
        print(md)


if __name__ == "__main__":
    main()
