# """
# @명령어: python color_change.py

# """



import zipfile
import os
import shutil
import numpy as np
from PIL import Image
import colorsys
import re

# ==========================================
# [설정] 이미지 색상(target_hex)을 지정
# ==========================================

input_file = "ori/miri_tem01.pptx"   # 원본 파일명
output_file = "result/miri_tem01_ch8_03.pptx"     # 저장될 파일명
# target_hex = "1A3C34"  # ch3 약간 형광 연두가 나와버렸음
# target_hex = "34495E"  # ch4 다크 슬레이트 블루 라는데, 좀더 파랑~한 바다 느낌으로 변했고 어느정도 맘에 듬.
# target_hex = "8D4004"  # ch5 구운 오렌지/벽돌색 (Burnt Orange/Brown) -> 주황색인데 너무 별로임
# target_hex = "431499"  # ch6 딥 퍼플 (Deep Purple) -> 보라색 계열로 바꿔봄 0.5 버전 꽤나이쁨! 연보라
# target_hex = "321266"  # ch7 딥 퍼플 (Deep Purple) -> 남색에 가까운 보라색 계열
target_hex = "66123B"  # ch8 딥 퍼플 (Deep Purple) -> 와인레드에 가까운 색상 0.8 너무 핫핑크다..
# 채도 조절 옵션 (1.0 = 원본 유지, 0.7 = 30% 더 차분하게/탁하게 만듦)
saturation_scale = 0.3  # 0.8 이 적당히 쨍하고 이쁜 것 같음!
# ==========================================

def hex_to_rgb(hex_str):
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def process_image_smart_shift(img_path, target_rgb, sat_scale):
    """
    이미지의 명암(Value)은 유지하고 색상(Hue)만 타겟으로 교체합니다.
    """
    try:
        img = Image.open(img_path).convert("RGBA")
        # numpy 배열로 변환 (속도 향상을 위해)
        arr = np.array(img, dtype=np.float32)
        
        # 0~255 값을 0~1로 정규화
        r, g, b, a = arr[:,:,0]/255.0, arr[:,:,1]/255.0, arr[:,:,2]/255.0, arr[:,:,3]/255.0
        
        # 1. 흰색 배경 투명화 처리 (RGB > 0.94 즉, 240 이상)
        white_mask = (r > 0.94) & (g > 0.94) & (b > 0.94)
        a[white_mask] = 0  # 투명도 0
        
        # ----------------------------------------------------
        # 2. RGB -> HSV 변환 (벡터 연산)
        # ----------------------------------------------------
        # HSV 변환 로직 구현 (matplotlib 등 외부 라이브러리 의존성 제거를 위해 직접 구현)
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        v = max_c
        d = max_c - min_c
        
        s = np.zeros_like(v)
        mask_d = d > 0
        s[mask_d] = d[mask_d] / max_c[mask_d]
        
        h = np.zeros_like(v)
        # Hue 계산
        mask_r = (max_c == r) & mask_d
        mask_g = (max_c == g) & mask_d
        mask_b = (max_c == b) & mask_d
        
        h[mask_r] = (g[mask_r] - b[mask_r]) / d[mask_r]
        h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / d[mask_g]
        h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / d[mask_b]
        
        h = (h / 6.0) % 1.0
        # ----------------------------------------------------

        # 3. 타겟 색상의 Hue, Saturation 추출
        tr, tg, tb = [x/255.0 for x in target_rgb]
        tr_h, tr_s, tr_v = colorsys.rgb_to_hsv(tr, tg, tb)

        # 4. '파란색 계열' 픽셀 탐지 (Hue 0.5 ~ 0.75 범위)
        # 너무 어둡거나(v<0.2) 채도가 없는(s<0.1) 영역은 색상이 아니므로 제외
        blue_mask = (h >= 0.50) & (h <= 0.75) & (s > 0.15) & (v > 0.2)
        
        # 5. 색상 교체 (핵심 로직)
        # (1) Hue: 타겟 색상으로 강제 변경
        h[blue_mask] = tr_h
        
        # (2) Saturation: 원본의 농도를 유지하되, 전체적으로 차분하게(sat_scale) 조절
        # 단, 타겟 색상이 아주 연하면 타겟 채도를 따라가도록 블렌딩
        s[blue_mask] = s[blue_mask] * sat_scale
        
        # (3) Value(명도): 건드리지 않음! -> 이것이 디테일을 살리는 핵심입니다.
        
        # ----------------------------------------------------
        # 6. HSV -> RGB 역변환
        # ----------------------------------------------------
        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        
        new_r = np.zeros_like(r)
        new_g = np.zeros_like(g)
        new_b = np.zeros_like(b)
        
        # Case 0
        mask = i == 0
        new_r[mask], new_g[mask], new_b[mask] = v[mask], t[mask], p[mask]
        # Case 1
        mask = i == 1
        new_r[mask], new_g[mask], new_b[mask] = q[mask], v[mask], p[mask]
        # Case 2
        mask = i == 2
        new_r[mask], new_g[mask], new_b[mask] = p[mask], v[mask], t[mask]
        # Case 3
        mask = i == 3
        new_r[mask], new_g[mask], new_b[mask] = p[mask], q[mask], v[mask]
        # Case 4
        mask = i == 4
        new_r[mask], new_g[mask], new_b[mask] = t[mask], p[mask], v[mask]
        # Case 5
        mask = i == 5
        new_r[mask], new_g[mask], new_b[mask] = v[mask], p[mask], q[mask]
        
        # 채널 합치기 (Alpha 채널 복구)
        new_arr = np.dstack((new_r, new_g, new_b, a)) * 255.0
        new_arr = np.clip(new_arr, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(new_arr)
        result_img.save(img_path)
        return True
        
    except Exception as e:
        print(f"이미지 처리 오류 {img_path}: {e}")
        return False

# ==========================================
# 메인 실행 부
# ==========================================

# 1. 임시 폴더 준비
temp_dir = "temp_pptx_smart"
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
    print("이미지 색상 변환 중... (시간이 조금 걸릴 수 있습니다)")
    count = 0
    for filename in os.listdir(media_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image_smart_shift(os.path.join(media_dir, filename), target_rgb_val, saturation_scale)
            count += 1
    print(f"{count}개의 이미지 변환 완료.")

# 4. 테마 XML (텍스트/도형) 처리 - 이건 기존 방식이 깔끔하므로 유지 (단색 치환)
theme_path = os.path.join(temp_dir, 'ppt', 'theme', 'theme1.xml')
if os.path.exists(theme_path):
    with open(theme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 테마는 그라데이션이 아니라 단색 설정이므로 Hex 코드만 교체해도 됨
    def replace_theme_color(match):
        h_code = match.group(1)
        r = int(h_code[:2], 16) / 255.0
        g = int(h_code[2:4], 16) / 255.0
        b = int(h_code[4:6], 16) / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # 파란색 계열이면 교체
        if (0.50 <= h <= 0.75) and (s > 0.4):
            return f'val="{target_hex}"'
        return match.group(0)

    new_content = re.sub(r'val="([0-9A-Fa-f]{6})"', replace_theme_color, content)
    with open(theme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

# 5. 재압축
with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zip_out:
    for foldername, subfolders, filenames in os.walk(temp_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            arcname = os.path.relpath(file_path, temp_dir)
            zip_out.write(file_path, arcname)

shutil.rmtree(temp_dir)
print(f"완료! '{output_file}' 파일을 확인해보세요.")