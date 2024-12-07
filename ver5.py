import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
import base64
import requests
from deep_translator import GoogleTranslator
import os
from dotenv import load_dotenv
# GPT-4o API 설정
GPT4O_API_URL = "https://api.openai.com/v1/chat/completions"
#GPT4O_API_KEY = "sk-svcacct-gTs7WGJV79tILrE7sVMTe3Q7qP-0qlSXYZCzSWy-lpL3wDmwM21mYEKwx3j8YbT3BlbkFJBG2Iu9xC7Cw5FH5_cp2c78TswAkA0ipItUltr09UkXNyHhxKGSiQhvy4uM6TkA"  # OpenAI API 키를 입력하세요.

# 환경 변수에서 API 키 가져오기 (Streamlit Secrets 사용)
#GPT4O_API_KEY = st.secrets["openai"]["api_key"]  # 비밀 키 불러오기
#GPT4O_API_KEY=os.getenv("sk-proj-1HYmZavxNlV7PYui14ugT3BlbkFJr0KwBq0QuNBPrkPyTbtk")

# Load API key based on environment
if "openai" in st.secrets:
    # Streamlit Cloud 환경
    GPT4O_API_KEY = st.secrets["openai"]["api_key"]
else:
    # 로컬 환경
    load_dotenv()  # .env 파일 로드
    GPT4O_API_KEY = os.getenv("OPENAI_API_KEY")

if GPT4O_API_KEY:
    st.success("API Key Loaded Successfully!")
else:
    st.error("Failed to load API Key.")
# 번역 함수
def translate_text(text, target_language):
    translated = GoogleTranslator(source='auto', target=target_language).translate(text)
    return translated
# Base64 인코딩 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 모델 로드
if "model" not in st.session_state:
    st.session_state.model = keras.models.load_model('model.h5')

# 언어 설정
if "language" not in st.session_state:
    st.session_state.language = "Korean"

# 언어 선택
st.sidebar.title("Language / 언어")
language_toggle = st.sidebar.radio(
    label="Select language / 언어를 선택하세요",
    options=["English", "Korean"],
    index=1,
    key="language_toggle"
)
# 언어 설정 반영
st.session_state.language = "English" if language_toggle == "English" else "Korean"

# 다국어 지원 텍스트
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
    "image_analysis": {
        "English": ":camera: Analysis of the uploaded image:",
        "Korean": ":camera: 업로드된 이미지 분석:",
    },
    "usage_guide": {
        "English": """
        ### **Usage Guide**
        1. Select your preferred language using the sidebar.
        2. Upload an image file (PNG or JPG format) to analyze if it is AI-generated or not.
        3. View the prediction result along with the uploaded image.
        """,
        "Korean": """
        ### **사용 가이드**
        1. 사이드바에서 언어를 선택하세요.
        2. 이미지 파일(PNG 또는 JPG 형식)을 업로드하여 AI 생성 여부를 분석하세요.
        3. 분석 결과와 업로드된 이미지를 확인하세요.
        """,
    },
    "links_header": {
        "English": ":page_with_curl: Additional Resources",
        "Korean": ":page_with_curl: 참고 자료",
    },
    "error_message": {
        "English": "Failed to process the image description. Please try again later.",
        "Korean": "이미지 설명 처리가 실패했습니다. 나중에 다시 시도해주세요.",
    }
}

# 제목 및 업로드 섹션
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

    # GPT-4o를 통한 이미지 설명 요청
    st.subheader(texts["image_analysis"][st.session_state.language])
    with open("uploaded_image.jpg", "wb") as f:
        f.write(image_file.getbuffer())
    base64_image = encode_image("uploaded_image.jpg")

    # GPT-4o API로 언어에 맞게 설명 요청
    prompt = "Describe this image."
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100  # 출력 길이를 제한
    }

    headers = {
        "Authorization": f"Bearer {GPT4O_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GPT4O_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        gpt_response = response.json()
        description = gpt_response['choices'][0]['message']['content']
        # 한국어로 번역
        if st.session_state.language == "Korean":
            translated_description = translate_text(description, "ko")
            st.markdown(f"**{translated_description}**")
        else:
            st.markdown(f"**{description}**")
    else:
        st.error(texts["error_message"][st.session_state.language])

# 가이드라인 섹션 추가
with st.expander("**GuideLine / 가이드라인**"):
    st.markdown(texts["usage_guide"][st.session_state.language])
st.divider()

# 참고 자료 섹션 추가
st.header(texts["links_header"][st.session_state.language])
st.markdown("* :memo: **From Team C-생성형 AI 프로젝트**")