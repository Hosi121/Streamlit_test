import numpy as np
from PIL import Image
import streamlit as st
from faiss_kneighbors import FaissKNeighbors

# init KNN and train
def init_knn(X_train, k=5):
    knn = FaissKNeighbors(k=k)
    knn.fit(X_train.reshape(-1, 28 * 28))
    return knn

# KNN result
def run_knn_classification(canvas_result, knn, Y_train, X_train):
    # get the image
    drawn_image = canvas_result.image_data
    drawn_image_gray = drawn_image[:, :, 3]  # use alphachannel

    # predict, and get result
    if np.sum(drawn_image_gray) > 0:  # execute only
        # resize the image
        drawn_image_gray = Image.fromarray(drawn_image_gray)
        resized_image = drawn_image_gray.resize((28, 28))
        resized_image = np.array(resized_image).reshape(1, -1)
        
        # classfy by KNN
        dists, indices = knn.predict(resized_image)
        votes = Y_train[indices.flatten()]

        if len(votes) > 0:
            prediction = np.bincount(votes).argmax()
            user_prediction = st.session_state.user_prediction
            if user_prediction == prediction:
                st.session_state.correct += 1
            st.session_state.attempts += 1

            # calculate accuracy
            accuracy = (st.session_state.correct / st.session_state.attempts) * 100

            # show the result
            st.write(f"あなたの予想: {user_prediction}, KNNによる判定結果: {prediction}")
            st.write(f"類似度上位の数字:")
            cols = st.columns(knn.k)
            for i, idx in enumerate(indices.flatten()):
                with cols[i]:
                    pred_image = Image.fromarray(X_train[idx])
                    pred_image = pred_image.resize((100, 100))
                    pred_image = np.array(pred_image)
                    st.image(pred_image, clamp=True, caption=f'ラベル = {Y_train[idx]}')
            st.metric(label="正解率", value=f"{accuracy:.2f}%")
