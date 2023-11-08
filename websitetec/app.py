import streamlit as st
from PIL import Image
import numpy as np

# utils.pyから必要な関数をインポート
from utils import load_mnist_data, load_explanations, load_css, load_model
# canvas.pyからキャンバス表示関数をインポート
from canvas import display_canvas

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

# ユーザーの予測を取得するラジオボタン
user_prediction = st.radio(
    "あなたの予測は何ですか？",
    options=list(range(10)),  # 0から9までの選択肢
    index=st.session_state.user_prediction  # デフォルトの選択肢
)

# 予測をセッション状態に保存
st.session_state.user_prediction = user_prediction

# 予測送信ボタン
if st.button('予想を送信'):
    st.write(f"送信された予測: {user_prediction}")
    
# Assuming you have already defined columns in your Streamlit layout
col1, col2, col3 = st.columns([2,5,3])

# Now you can pass one of these columns to the display_canvas function
canvas_result = display_canvas(col2)

# CNNを使用して予測する関数
def predict_with_cnn(img, model):
    # 画像をモデルの入力サイズにリサイズし、正規化する
    img = img.reshape((1, 28, 28, 1)).astype('float32') / 255
    
    # 予測を行う
    prediction = model.predict(img)
    return np.argmax(prediction), max(prediction[0])

# キャンバスに何か描かれていれば予測を行う
if canvas_result.image_data is not None:
    # 画像データの前処理
    img = canvas_result.image_data[:, :, 3]  # アルファチャネルを取得
    img = Image.fromarray(img).resize((28, 28)).convert('L')
    img = np.array(img)

    # CNNモデルで予測
    prediction, confidence = predict_with_cnn(img, cnn_model)
    st.write(f"判定結果: {prediction} (信頼度: {confidence*100:.2f}点)")

# サイドバーの表示
with st.sidebar:
    explanations_md = load_explanations('explanations.md')
    st.markdown(explanations_md, unsafe_allow_html=True)
