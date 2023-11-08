import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# utils.pyから必要な関数をインポート
# 以下の関数は、実際のファイルとその内容に基づいて適宜調整してください。
from utils import load_css, load_explanations

# スタイルシートの読み込み
load_css('style.css')

# モデルの読み込み
cnn_model = load_model('my_model.keras')

# アプリケーションのタイトル
st.title("手書き数字認識アプリ")
st.write("マウスで数字を描いて、予想を選んでから「予想を送信」を押してください。")

# セッション状態の初期化
if 'user_prediction' not in st.session_state:
    st.session_state.user_prediction = 0
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
    st.session_state.correct = 0

# ユーザーの予測を取得するラジオボタン
user_prediction = st.radio(
    "あなたの予測は何ですか？",
    options=list(range(10)),  # 0から9までの選択肢
    index=st.session_state.user_prediction  # デフォルトの選択肢
)

# 予測をセッション状態に保存
st.session_state.user_prediction = user_prediction

# キャンバスの設定
col1, col2, col3 = st.columns([2,5,3])
canvas_result = st_canvas(
    stroke_width=20,
    update_streamlit=False,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key="canvas",
    column=col2  # キャンバスを表示する列
)

# 予測送信ボタン
if st.button('予想を送信'):
    # 画像データの前処理
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 3]  # アルファチャネルを取得
        img = Image.fromarray(img).resize((28, 28)).convert('L')
        img = np.array(img).reshape((1, 28, 28, 1)).astype('float32') / 255

        # CNNモデルで予測
        prediction = cnn_model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 予測結果の表示
        st.write(f"判定結果: {predicted_class} (信頼度: {confidence*100:.2f}%)")
        
        # 正解数と試行回数の更新
        st.session_state.attempts += 1
        if user_prediction == predicted_class:
            st.session_state.correct += 1
        
        # 正解率の計算と表示
        accuracy = (st.session_state.correct / st.session_state.attempts) * 100
        st.write(f"正解率: {accuracy:.2f}%")
    else:
        st.write("予測するためには、まずキャンバスに数字を描いてください。")

    # ユーザーの予測の表示
    st.write(f"送信された予測: {user_prediction}")

# サイドバーの表示
with st.sidebar:
    explanations_md = load_explanations('explanations.md')
    st.markdown(explanations_md, unsafe_allow_html=True)
