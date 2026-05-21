import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import easyocr
import io

# [추가] 회사명 리스트 정의
COMPANY_LIST = [
    "Webon Automotive", "KCC오토모빌", "JL Motors", "Hyosung Premier Motors",
    "Hanyoung Motors", "Entire Motors", "Chunil Automobile", "Aju Networks"
]

st.set_page_config(page_title="PDI Quota OCR 자동 분할 버전", layout="wide")

if 'all_rows_data' not in st.session_state: st.session_state.all_rows_data = []
if 'split_images' not in st.session_state: st.session_state.split_images = []

st.title("🎯 PDI Quota OCR (자동 행 분할 모드)")

@st.cache_resource
def load_easy_ocr(): return easyocr.Reader(['en'], gpu=False)
reader = load_easy_ocr()

# 1. 전체 표 이미지 업로드 및 자동 분할
uploaded_file = st.file_uploader("전체 표 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file and not st.session_state.split_images:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    h, w, _ = img_np.shape
    row_h = h // 8
    st.session_state.split_images = [img_np[i*row_h:(i+1)*row_h, :] for i in range(8)]
    st.rerun()

# 2. 행별 검토 로직
if st.session_state.split_images:
    idx = st.slider("검토할 행 선택", 0, 7, 0)
    img_np = st.session_state.split_images[idx]
    
    # [추가] 현재 선택된 행의 회사명 표시
    st.info(f"📍 현재 검수 중인 회사: **{COMPANY_LIST[idx]}**")
    
    h, w, _ = img_np.shape
    img_pil = Image.fromarray(img_np)
    st.image(img_pil.resize((w*2, h*2)), use_container_width=True)

    has_total = st.checkbox("Total 포함", value=True, key=f"total_{idx}")
    
    if st.button("이 행 OCR 실행"):
        with st.spinner("판독 중..."):
            # --- [기존 로직 시작] ---
            gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.adaptiveThreshold(gray_full, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.5)))
            detect_vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            contours, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_x = int(w * 0.208)
            x_coords = [cv2.boundingRect(c)[0] for c in contours if int(w * 0.10) < cv2.boundingRect(c)[0] < int(w * 0.40)]
            if x_coords: start_x = min(x_coords)
            
            end_x = int(w * 0.985)
            col_width = (end_x - start_x) / (33 if has_total else 32)
            
            digits = []
            for i in range(1, 32):
                cell = img_np[int(h*0.16):int(h*0.84), int(start_x + (i*col_width) + 2):int(start_x + ((i+1)*col_width) - 2)]
                gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=6.5, tileGridSize=(2,2))
                enhanced = clahe.apply(gray)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.mean(thresh) < 127: thresh = cv2.bitwise_not(thresh)
                final_cell = cv2.resize(thresh, (0,0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                result = reader.readtext(final_cell, allowlist='0123456789', paragraph=False)
                digits.append(result[0][1].strip() if result else "0")
            # --- [기존 로직 끝] ---
            st.session_state[f"digits_{idx}"] = digits

    if f"digits_{idx}" in st.session_state:
        df = pd.DataFrame({i: [st.session_state[f"digits_{idx}"][i-1]] for i in range(1, 32)})
        edited_df = st.data_editor(df, use_container_width=True)
        
        if st.button("💾 이 행 데이터 세트에 추가"):
            # [수정] 데이터 저장 시 회사명 포함
            row_dict = {"Retailer Name": COMPANY_LIST[idx]}
            row_dict.update(edited_df.iloc[0].to_dict())
            st.session_state.all_rows_data.append(row_dict)
            st.success(f"{COMPANY_LIST[idx]} 저장 완료!")

# 3. 누적 데이터 처리
if st.session_state.all_rows_data:
    st.divider()
    all_df = pd.DataFrame(st.session_state.all_rows_data)
    st.dataframe(all_df, use_container_width=True)
    if st.button("🚀 전체 엑셀 다운로드"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_df.to_excel(writer, index=False)
        st.download_button("📥 최종 엑셀 저장", data=output.getvalue(), file_name="final_result.xlsx")
