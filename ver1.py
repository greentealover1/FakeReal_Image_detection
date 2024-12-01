import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# 모델 로드
if "model" not in st.session_state:
    st.session_state.model = keras.models.load_model('model.h5')

#언어 설정
if "language" not in st.session_state:
    st.session_state.language = "Korean"

#언어 선택 토글
st.sidebar.title("Language / 언어")
language_toggle = st.sidebar.radio(
    label="Select language / 언어를 선택하세요",
    options=["English", "Korean"],
    index=1,
    key="language_toggle"
)
st.session_state.language = "English" if language_toggle == "English" else "Korean"

#다국어 지원 텍스트
texts = {
    "title": {
        "English": ":desktop_computer: FAKE Image Detection Model",
        "Korean": ":desktop_computer: FAKE 이미지 판별모델",
    },
    "upload_header": {
        "English": ":envelope: Image Upload",
        "Korean": ":envelope: 이미지 업로드",
    },
    "upload_label": {
        "English": "Upload the image to verify.",
        "Korean": "확인할 이미지를 업로드하세요.",
    },
    "ai_generated": {
        "English": ":robot_face: **{confidence:.2f}%** probability that the image is AI-generated.",
        "Korean": ":robot_face: **{confidence:.2f}%**의 확률로 AI 생성 이미지입니다.",
    },
    "not_ai_generated": {
        "English": ":male-artist: **{confidence:.2f}%** probability that the image is not AI-generated.",
        "Korean": ":male-artist: **{confidence:.2f}%**의 확률로 AI 생성 이미지가 아닙니다.",
    },
    "usage_guide": {
        "English": """
        **Usage Guide**
        1. Select the language.
        2. Drag and drop the image file in the 'Image Upload' section.
        """,
        "Korean": """
        **사용 가이드**
        1. 언어를 선택하세요.
        2. '이미지 업로드' 섹션에 이미지 파일을 드래그하여 업로드하세요.
        """,
    },
    "links_header": {
        "English": ":page_with_curl: Additional Resources",
        "Korean": ":page_with_curl: 같이 보기",
    }
}

#session state 초기화
if "model" not in st.session_state:
    st.session_state.model = keras.models.load_model('model.h5')

#제목 및 업로드 섹션
st.title(texts["title"][st.session_state.language])
st.divider()
st.header(texts["upload_header"][st.session_state.language])
image_file = st.file_uploader(
    label=texts["upload_label"][st.session_state.language],
    type=["png", "jpg"],
    accept_multiple_files=False,
    key="image_uploader"
)
st.divider()

if image_file is not None:
    raw_image = Image.open(image_file)
    rgb_image = raw_image.convert('RGB')
    mini_image = rgb_image.resize(size=(32, 32))

    image = np.array(mini_image, dtype='float')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = st.session_state.model.predict(image)
    confidence = round(float(prediction[0][0]) * 100, 2)

    is_generated_by_ai = confidence < 50.0
    if is_generated_by_ai:
        st.header(texts["ai_generated"][st.session_state.language].format(confidence=100 - confidence))
    else:
        st.header(texts["not_ai_generated"][st.session_state.language].format(confidence=confidence))

    st.image(image_file)
    st.divider()

#가이드라인 추가 (토글 형식)
with st.expander("**GuideLine / 가이드라인**"):
    st.markdown(texts["usage_guide"][st.session_state.language])
st.divider()

#같이 보기 섹션
st.header(texts["links_header"][st.session_state.language])
st.markdown("* :memo: From Team C-생성형 Ai프로젝트")
st.markdown("* :memo: [Google Colaboratory](https://colab.research.google.com/drive/1PL2vC3NOWrJgX7ghpu_MAxy9186owuSK?usp=sharing)")
st.markdown("* :book: [Kaggle/CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)")
st.markdown("* :book: [Kaggle/CIFAKE-midjourney](https://www.kaggle.com/datasets/mariammarioma/midjourney-cifake-inspired)")
