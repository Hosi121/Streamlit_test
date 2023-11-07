import streamlit as st
from streamlit_drawable_canvas import st_canvas

def display_canvas(col):
    with col:
        # マウスで文字を描画するキャンバスを作成
        return st_canvas(
            stroke_width=20,
            update_streamlit=False,
            height=200,
            width=200,
            drawing_mode='freedraw',
            key="canvas"
        )
