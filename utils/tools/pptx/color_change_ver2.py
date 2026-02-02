# """
# @명령어: python color_change_ver2.py
# color_change.py 의 개선판. 텍스트도 원하는 컬러로 수정한다!
# """

import zipfile
import os
import shutil
import numpy as np
from PIL import Image
import colorsys
import re

# ==========================================
# [설정] 이미지 색상(target_hex)과 텍스트 색상(text_target_hex)을 각각 지정 가능
# ==========================================

input_file = "ori/miri_tem01.pptx"   # 원본 파일명
output_file = "result/miri_tem01_final_ch4_03_textadd.pptx"     # 저장될 파일명

# 변경할 목표 색상 (이미지 & 텍스트 공통 적용)
# target_hex = "66123B"  # 와인레드/딥퍼플 계열 final_ch2_03
# target_hex = "221833"  # 딥퍼플/네이비 계열 final_ch3_03
target_hex = "6F58AD"  # 라벤더 퍼플 계열 final_ch4_03

# (옵션) 텍스트는 다른 색으로 하고 싶다면 여기서 변경 (기본값: target_hex와 동일)
# text_target_hex = target_hex 
text_target_hex = "30244F" 

# 채도 조절 옵션 (1.0 = 원본 유지, 0.7 = 30% 더 차분하게/탁하게 만듦)
saturation_scale = 0.3 
# ==========================================

def hex_to_rgb(hex_str):
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def process_image_smart_shift(img_path, target_rgb, sat_scale):
    """
    이미지 처리 로직 (이전과 동일하게 유지)
    """
    try:
        img = Image.open(img_path).convert("RGBA")
        arr = np.array(img, dtype=np.float32)
        
        r, g, b, a = arr[:,:,0]/255.0, arr[:,:,1]/255.0, arr[:,:,2]/255.0, arr[:,:,3]/255.0
        
        white_mask = (r > 0.94) & (g > 0.94) & (b > 0.94)
        a[white_mask] = 0 
        
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        v = max_c
        d = max_c - min_c
        
        s = np.zeros_like(v)
        mask_d = d > 0
        s[mask_d] = d[mask_d] / max_c[mask_d]
        
        h = np.zeros_like(v)
        mask_r = (max_c == r) & mask_d
        mask_g = (max_c == g) & mask_d
        mask_b = (max_c == b) & mask_d
        
        h[mask_r] = (g[mask_r] - b[mask_r]) / d[mask_r]
        h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / d[mask_g]
        h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / d[mask_b]
        
        h = (h / 6.0) % 1.0

        tr, tg, tb = [x/255.0 for x in target_rgb]
        tr_h, tr_s, tr_v = colorsys.rgb_to_hsv(tr, tg, tb)

        blue_mask = (h >= 0.50) & (h <= 0.75) & (s > 0.15) & (v > 0.2)
        
        h[blue_mask] = tr_h
        s[blue_mask] = s[blue_mask] * sat_scale
        
        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        
        new_r = np.zeros_like(r)
        new_g = np.zeros_like(g)
        new_b = np.zeros_like(b)
        
        mask = i == 0; new_r[mask], new_g[mask], new_b[mask] = v[mask], t[mask], p[mask]
        mask = i == 1; new_r[mask], new_g[mask], new_b[mask] = q[mask], v[mask], p[mask]
        mask = i == 2; new_r[mask], new_g[mask], new_b[mask] = p[mask], v[mask], t[mask]
        mask = i == 3; new_r[mask], new_g[mask], new_b[mask] = p[mask], q[mask], v[mask]
        mask = i == 4; new_r[mask], new_g[mask], new_b[mask] = t[mask], p[mask], v[mask]
        mask = i == 5; new_r[mask], new_g[mask], new_b[mask] = v[mask], p[mask], q[mask]
        
        new_arr = np.dstack((new_r, new_g, new_b, a)) * 255.0
        new_arr = np.clip(new_arr, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(new_arr)
        result_img.save(img_path)
        return True
        
    except Exception as e:
        print(f"이미지 처리 오류 {img_path}: {e}")
        return False

def replace_color_in_xml_safe(file_path, target_color_hex):
    """
    [핵심 수정] XML 파일에서 확실한 색상 태그(srgbClr)만 찾아서 변경합니다.
    """
    if not os.path.exists(file_path):
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 정규식 수정: <a:srgbClr ... val="XXXXXX" ...> 형태만 찾습니다.
    # Group 1: 태그 앞부분 (<a:srgbClr ... val=")
    # Group 2: 색상 코드 (XXXXXX)
    # Group 3: 뒷부분 (")
    pattern = r'(<[a-zA-Z0-9]+:srgbClr[^>]*\sval=")([0-9A-Fa-f]{6})(")'

    def replace_match(match):
        prefix = match.group(1)
        h_code = match.group(2)
        suffix = match.group(3)
        
        try:
            r = int(h_code[:2], 16) / 255.0
            g = int(h_code[2:4], 16) / 255.0
            b = int(h_code[4:6], 16) / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            
            # 파란색 계열이고 채도가 어느정도 있는 경우만 교체
            if (0.50 <= h <= 0.75) and (s > 0.4):
                return f'{prefix}{target_color_hex}{suffix}'
        except:
            pass
        return match.group(0)

    new_content = re.sub(pattern, replace_match, content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

# ==========================================
# 메인 실행 부
# ==========================================

# 1. 임시 폴더 준비
temp_dir = "temp_pptx_fixed"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

# 2. 압축 해제
with zipfile.ZipFile(input_file, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# 3. 이미지 처리
target_rgb_val = hex_to_rgb(target_hex)
media_dir = os.path.join(temp_dir, 'ppt', 'media')

if os.path.exists(media_dir):
    print("이미지 색상 변환 중...")
    count = 0
    for filename in os.listdir(media_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image_smart_shift(os.path.join(media_dir, filename), target_rgb_val, saturation_scale)
            count += 1
    print(f"{count}개의 이미지 변환 완료.")

# 4. XML 텍스트/테마 색상 변경 (안전 모드 적용)
target_dirs = [
    os.path.join(temp_dir, 'ppt', 'theme'),
    os.path.join(temp_dir, 'ppt', 'slides'),
    os.path.join(temp_dir, 'ppt', 'slideMasters'),
    os.path.join(temp_dir, 'ppt', 'slideLayouts')
]

print("텍스트 및 테마 색상 정밀 보정 중...")
xml_count = 0
for directory in target_dirs:
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".xml"):
                process_path = os.path.join(directory, filename)
                replace_color_in_xml_safe(process_path, text_target_hex)
                xml_count += 1
print(f"{xml_count}개의 XML 파일 처리 완료.")

# 5. 재압축
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zip_out:
    for foldername, subfolders, filenames in os.walk(temp_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            arcname = os.path.relpath(file_path, temp_dir)
            zip_out.write(file_path, arcname)

shutil.rmtree(temp_dir)
print(f"완료! 오류 없는 파일이 생성되었습니다: '{output_file}'")