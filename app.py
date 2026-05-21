import streamlit as st
import pandas as pd
from PIL import Image, ImageGrab
import numpy as np
import cv2
import easyocr
import io

st.set_page_config(page_title="최종 완성형 스마트 OCR", layout="wide")

if 'all_rows_data' not in st.session_state:
    st.session_state.all_rows_data = []

st.title("PDI Quota OCR")

@st.cache_resource
def load_easy_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_easy_ocr()

# 1. 이미지 선택 영역
tab1, tab2 = st.tabs(["📂 파일 업로드", "📋 클립보드에서 붙여넣기"])
image_to_process = None

with tab1:
    uploaded_file = st.file_uploader("이미지를 선택하세요", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)

with tab2:
    if st.button("📋 클립보드에서 불러오기"):
        clipboard_img = ImageGrab.grabclipboard()
        if clipboard_img:
            image_to_process = clipboard_img
        else:
            st.error("클립보드에 이미지가 없습니다.")

# 2. 이미지가 선택되었을 때만 하단 로직 실행
if image_to_process:
    img_np = np.array(image_to_process)
    h, w, _ = img_np.shape
    
    # 이미지 확대 표시
    image_scaled = image_to_process.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    st.subheader("📷 확대된 원본 이미지 (1:1 대조용)")
    st.image(image_scaled, use_container_width=True)

    has_total = st.checkbox("이미지 맨 우측 끝에 'Total(합계)' 칸이 포함되어 있습니까?", value=True)

    if st.button("🚀 판독 시작"):
        with st.spinner("이미지 최적화 및 판독 중..."):
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
                cell_start = int(start_x + (i * col_width))
                cell_end = int(start_x + ((i + 1) * col_width))
                cell = img_np[int(h*0.16):int(h*0.84), int(cell_start + ((cell_end-cell_start)*0.06)):int(cell_end - ((cell_end-cell_start)*0.06))]
                
                gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=6.5, tileGridSize=(2,2))
                enhanced = clahe.apply(gray)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.mean(thresh) < 127: thresh = cv2.bitwise_not(thresh)
                final_cell = cv2.resize(thresh, (0,0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                result = reader.readtext(final_cell, allowlist='0123456789', paragraph=False)
                digits.append(result[0][1].strip() if result else "0")
            
            total_val = "0"
            if has_total:
                total_start = int(start_x + (32 * col_width))
                total_result = reader.readtext(cv2.cvtColor(img_np[:, total_start:end_x], cv2.COLOR_RGB2GRAY), allowlist='0123456789')
                if total_result: total_val = total_result[0][1].strip()

            # 결과를 세션 상태에 저장하여 판독 후에도 화면 유지
            st.session_state['last_digits'] = digits
            st.session_state['last_total'] = total_val

    # 판독 결과 검수 화면
    if 'last_digits' in st.session_state:
        st.subheader("📊 단일 행 데이터 검수")
        matrix = {i: [st.session_state['last_digits'][i-1]] for i in range(1, 32)}
        if has_total: matrix["Total"] = [st.session_state['last_total']]
        df = pd.DataFrame(matrix)
        edited_df = st.data_editor(df, use_container_width=True)
        
        if st.button("💾 이 행을 전체 데이터 세트에 추가하기"):
            st.session_state.all_rows_data.append(edited_df.iloc[0].to_dict())
            st.success("데이터 세트에 추가되었습니다!")

# 3. 누적 데이터 처리 (기존과 동일)
if st.session_state.all_rows_data:
    st.divider()
    st.subheader(f"📊 누적 데이터 ({len(st.session_state.all_rows_data)}개 행)")
    all_df = pd.DataFrame(st.session_state.all_rows_data)
    st.dataframe(all_df, use_container_width=True)
    # ... (다운로드 버튼 로직)

# 누적된 데이터 최종 처리
if st.session_state.all_rows_data:
    st.divider()
    st.subheader(f"📊 누적 데이터 ({len(st.session_state.all_rows_data)}개 행)")
    all_df = pd.DataFrame(st.session_state.all_rows_data)
    st.dataframe(all_df, use_container_width=True)

    

    if st.button("🚀 8개 행 전체 엑셀 다운로드"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_df.to_excel(writer, index=False)
        st.download_button("📥 최종 엑셀 파일 저장", data=output.getvalue(), file_name="final_merged_result.xlsx")

        

    if st.button("🗑 데이터 전체 초기화"):
        st.session_state.all_rows_data = []
        st.rerun() 
