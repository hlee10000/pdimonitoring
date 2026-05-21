import streamlit as st

import pandas as pd

from PIL import Image

import numpy as np

import cv2

import easyocr

import io



st.set_page_config(page_title="최종 완성형 스마트 OCR", layout="wide")



# 데이터 누적용 세션 상태 초기화

if 'all_rows_data' not in st.session_state:

    st.session_state.all_rows_data = []



st.title("PDI Quota OCR")

st.write("사용방법: (1) 리테일러사별 행 (total 포함)을 캡쳐하여 업로드 및 OCR 인식 (2) 업로드 이미지와 대조하여 잘못 인식된 개수 고치기 (3) 검토 완료한 행은 전체 데이터 세트에 추가하기 (4) 8개사 반복 후 전체 데이터 세트 엑셀로 추출하기")



@st.cache_resource

def load_easy_ocr():

    return easyocr.Reader(['en'], gpu=False)



reader = load_easy_ocr()



uploaded_files = st.file_uploader("이미지를 여러 장 한꺼번에 선택하세요", type=["png", "jpg", "jpeg"], accept_multiple_files=True)



if uploaded_files:

    # 루프를 돌며 처리합니다

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file)

        img_np = np.array(image)

    

    # [핵심 추가] 이미지 크기 2배 확대 (가독성 확보)

    w_orig, h_orig = image.size

    image_scaled = image.resize((w_orig * 2, h_orig * 2), Image.Resampling.LANCZOS)

    

    img_np = np.array(image)

    h, w, _ = img_np.shape



    st.subheader("📷 확대된 원본 이미지 (1:1 대조용)")

    st.image(image_scaled, use_container_width=True)



    has_total = st.checkbox("이미지 맨 우측 끝에 'Total(합계)' 칸이 포함되어 있습니까?", value=True)



    with st.spinner("이미지 최적화 및 판독 중..."):

        # 기존 정밀 판독 로직 유지

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



    st.subheader("📊 단일 행 데이터 검수")

    matrix = {i: [digits[i-1]] for i in range(1, 32)}

    if has_total: matrix["Total"] = [total_val]

    df = pd.DataFrame(matrix)

    edited_df = st.data_editor(df, use_container_width=True)

    

    if st.button("💾 이 행을 전체 데이터 세트에 추가하기"):

        st.session_state.all_rows_data.append(edited_df.iloc[0].to_dict())

        st.success(f"{len(st.session_state.all_rows_data)}번째 행이 저장되었습니다!")



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
