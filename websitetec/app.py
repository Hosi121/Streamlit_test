import streamlit as st
from utils import load_mnist_data, load_explanations
from canvas import display_canvas
from knn_module import init_knn, run_knn_classification

load_css('styles.css')
X_train, Y_train = load_mnist_data()

# Appilication Start
st.title("手書き数字認識アプリ")
st.write("マウスで数字を描いて、予想を選んでから「予想を送信」を押してください。")

# init and set knn models
knn = init_knn(X_train, k=5)

if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
    st.session_state.correct = 0

# Create radio bottuns
if 'user_prediction' not in st.session_state:
    st.session_state.user_prediction = 0  # set init

user_prediction = st.radio(
    "あなたの予測は何ですか？",
    options=list(range(10)),  # select to 0 to 9
    index=st.session_state.user_prediction  # set the answer as default
)

st.session_state.user_prediction = user_prediction

if st.button('予想を送信'):
    # ここに予測を処理するコードを追加
    st.session_state.attempts += 1
    # 予測が正しいかどうかをチェックするロジックをここに追加
    # 例: if user_prediction == correct_label:
    #     st.session_state.correct += 1
    st.write(f"送信された予測: {user_prediction}")

# settings layout
col1, col2, col3 = st.columns([2,5,3])

# display canvas
canvas_result = display_canvas(col2)

# classfy by KNN
if canvas_result.image_data is not None:
    run_knn_classification(canvas_result, knn, Y_train, X_train)

# display sidebar
with st.sidebar:
    explanations_md = load_explanations('explanations.md')
    st.markdown(explanations_md, unsafe_allow_html=True)
