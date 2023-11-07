import faiss
import numpy as np
from PIL import Image
from keras.datasets import mnist
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from faiss_kneighbors import FaissKNeighbors

K_NUM = 5

_, (X_test, y_test) = mnist.load_data()
knn = FaissKNeighbors(k=K_NUM)
knn.fit(X_test.reshape(-1, 28*28))

with open('explanations.md', 'r', encoding='utf-8') as file:
    explanations_md = file.read()

with st.sidebar:
    st.markdown(explanations_md, unsafe_allow_html=True)


# Streamlitアプリケーションの開始
st.title("手書き数字認識アプリ")
st.write("マウスで数字を描いて、予想を選んでから「予想を送信」を押してください。")


# 正解記録を保持するためのセッションステートを初期化
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
    st.session_state.correct = 0

# キャンバスを中央に配置するための列を作成
col1, col2, col3 = st.columns([2,5,3])
with col2:
    # マウスで文字を描画するキャンバスを作成
    canvas_result = st_canvas(
        stroke_width=20,
        update_streamlit=False,
        height=200,
        width=200,
        drawing_mode='freedraw',
        key="canvas"
    )

# ユーザーの予想を中央に配置
if canvas_result.image_data is not None:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.write("あなたが描いた数字の予想は何ですか？")
        prediction_options = list(range(10))
        user_prediction = st.radio("選択してください", prediction_options, key="prediction")

    # 予想を送信するボタンを中央に配置
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button('予想を送信', key="submit"):
            # 画像の処理とKNNでの予測
            drawn_image = canvas_result.image_data
            drawn_image_gray = drawn_image[:, :, 3]
            if np.sum(drawn_image_gray) > 0:
                drawn_image_gray = Image.fromarray(drawn_image_gray)
                resized_image = drawn_image_gray.resize((28, 28))
                resized_image = np.array(resized_image).reshape(1, -1)

                # KNNで分類
                dists, indexs = knn.predict(resized_image)
                votes = y_test[indexs.flatten()]
                predictions = np.bincount(votes).argmax()

                # 予想とKNNの結果を表示
                st.write(f"あなたの予想: {user_prediction}, KNNによる判定結果: {predictions}")
                st.write(f"類似度上位{K_NUM}の数字:")
                cols = st.columns(K_NUM)
                for i, idx in enumerate(indexs.flatten()):
                    with cols[i]:
                        pred_image = Image.fromarray(X_test[idx])
                        pred_image = pred_image.resize((100, 100))
                        pred_image = np.array(pred_image)
                        st.image(pred_image, clamp=True, caption=f'ラベル = {y_test[idx]}')
                # 正解の判定と記録
                if predictions == user_prediction:
                    st.session_state.correct += 1
                st.session_state.attempts += 1
                # 正解率の計算
                accuracy = (st.session_state.correct / st.session_state.attempts) * 100

                # 正解率をバーチャートで表示
                st.bar_chart({"Accuracy": [accuracy]})
    # 正解率を表示するカラム
    with col2:
        if 'attempts' not in st.session_state:
            st.session_state.attempts = 0
            st.session_state.correct = 0
        if st.session_state.attempts > 0:
            accuracy = (st.session_state.correct / st.session_state.attempts) * 100
            st.metric(label="正解率", value=f"{accuracy:.2f}%")

    # スコア表示用のプレースホルダーを作成
    score_placeholder = st.empty()

    # 正解率をバーチャートで表示
    if 'attempts' in st.session_state and st.session_state.attempts > 0:
        accuracy = (st.session_state.correct / st.session_state.attempts) * 100
        score_placeholder.bar_chart({"Accuracy": [accuracy]})
