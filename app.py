import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import easyocr
import io

st.set_page_config(page_title="최종 완성형 스마트 OCR", layout="wide")

if 'all_rows_data' not in st.session_state: st.session_state.all_rows_data = []
if 'ocr_results' not in st.session_state: st.session_state.ocr_results = {}

st.title("PDI Quota OCR")

@st.cache_resource
def load_easy_ocr(): return easyocr.Reader(['en'], gpu=False)
reader = load_easy_ocr()

uploaded_files = st.file_uploader("이미지 선택", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        # [구조 변경] 판독 로직을 이 조건문 안으로 완벽히 가두었습니다. (OCR 로직은 건드리지 않음)
        if idx not in st.session_state.ocr_results:
            with st.spinner(f"{idx+1}번째 이미지 판독 중..."):
                image = Image.open(uploaded_file)
                img_np = np.array(image)
                h, w, _ = img_np.shape
                
                # 원본 이미지를 세션에 저장
                st.session_state.ocr_results[idx] = {"image": image, "digits": None, "total": None}
                
                # (기존 OCR 로직 시작)
                gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.adaptiveThreshold(gray_full, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.5)))
                detect_vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
                contours, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                start_x = int(w * 0.208)
                x_coords = [cv2.boundingRect(c)[0] for c in contours if int(w * 0.10) < cv2.boundingRect(c)[0] < int(w * 0.40)]
                if x_coords: start_x = min(x_coords)
                end_x = int(w * 0.985)
                has_total = True
                col_width = (end_x - start_x) / 33
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
                total_start = int(start_x + (32 * col_width))
                total_result = reader.readtext(cv2.cvtColor(img_np[:, total_start:end_x], cv2.COLOR_RGB2GRAY), allowlist='0123456789')
                if total_result: total_val = total_result[0][1].strip()
                # (기존 OCR 로직 끝)
                
                st.session_state.ocr_results[idx]["digits"] = digits
                st.session_state.ocr_results[idx]["total"] = total_val

        # 검수 화면 (OCR 로직 외부)
        res = st.session_state.ocr_results[idx]
        w_orig, h_orig = res["image"].size
        st.image(res["image"].resize((w_orig * 2, h_orig * 2), Image.Resampling.LANCZOS), use_container_width=True)
        
        matrix = {i: [res["digits"][i-1]] for i in range(1, 32)}
        matrix["Total"] = [res["total"]]
        edited_df = st.data_editor(pd.DataFrame(matrix), use_container_width=True, key=f"edit_{idx}")
        
        if st.button(f"💾 {idx+1}번째 행 추가하기", key=f"btn_{idx}"):
            st.session_state.all_rows_data.append(edited_df.iloc[0].to_dict())
            st.success("저장 완료!")

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
