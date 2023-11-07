import numpy as np
from PIL import Image
import streamlit as st
from faiss_kneighbors import FaissKNeighbors

# KNN 分類器の初期化とトレーニング
def init_knn(X_train, k=5):
    knn = FaissKNeighbors(k=k)
    knn.fit(X_train.reshape(-1, 28 * 28))
    return knn

# KNN による分類と結果の表示
def run_knn_classification(canvas_result, knn, y_test):
    # 描画された画像を取得
    drawn_image = canvas_result.image_data
    drawn_image_gray = drawn_image[:, :, 3]  # アルファチャンネルを使用

    # 予測を行い、結果を表示する
    if np.sum(drawn_image_gray) > 0:  # 描画された内容がある場合のみ実行
        # 画像のリサイズ
        drawn_image_gray = Image.fromarray(drawn_image_gray)
        resized_image = drawn_image_gray.resize((28, 28))
        resized_image = np.array(resized_image).reshape(1, -1)

        # KNN で分類
        dists, indices = knn.predict(resized_image)
        votes = y_test[indices.flatten()]
        prediction = np.bincount(votes).argmax()

        # 予測結果と正解記録の更新
        user_prediction = st.session_state.user_prediction
        if user_prediction == prediction:
            st.session_state.correct += 1
        st.session_state.attempts += 1

        # 正解率の計算
        accuracy = (st.session_state.correct / st.session_state.attempts) * 100

        # 結果の表示
        st.write(f"あなたの予想: {user_prediction}, KNNによる判定結果: {prediction}")
        st.write(f"類似度上位の数字:")
        cols = st.columns(knn.k)
        for i, idx in enumerate(indices.flatten()):
            with cols[i]:
                pred_image = Image.fromarray(y_test[idx])
                pred_image = pred_image.resize((100, 100))
                pred_image = np.array(pred_image)
                st.image(pred_image, clamp=True, caption=f'ラベル = {y_test[idx]}')

        # 正解率の表示
        st.metric(label="正解率", value=f"{accuracy:.2f}%")
