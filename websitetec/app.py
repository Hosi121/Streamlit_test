import streamlit as st
from utils import load_mnist_data, load_explanations
from canvas import display_canvas
from knn_module import init_knn, run_knn_classification

X_train, Y_train = load_mnist_data()

# Streamlitアプリケーションの開始
st.title("手書き数字認識アプリ")
st.write("マウスで数字を描いて、予想を選んでから「予想を送信」を押してください。")

# KNN モデルの初期化とトレーニング
knn = init_knn(X_train, k=5)

if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
    st.session_state.correct = 0

# ユーザーの予測を選択するためのラジオボタンを作成
if 'user_prediction' not in st.session_state:
    st.session_state.user_prediction = 0  # 初期値を設定

user_prediction = st.radio(
    "あなたの予測は何ですか？",
    options=list(range(10)),  # 0から9までの数字を選択肢とする
    index=st.session_state.user_prediction  # 現在の予測値をデフォルトとして設定
)

st.session_state.user_prediction = user_prediction

# 画面レイアウトの設定
col1, col2, col3 = st.columns([2,5,3])

# キャンバスの表示
canvas_result = display_canvas(col2)

# KNN による分類
if canvas_result.image_data is not None:
    run_knn_classification(canvas_result, knn, Y_train, X_train)

# サイドバーの表示
with st.sidebar:
    explanations_md = load_explanations('explanations.md')
    st.markdown(explanations_md, unsafe_allow_html=True)
