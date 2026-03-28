from __future__ import annotations

import gradio as gr

from api.api import api_run, get_app
from prediction import predict_sentiment
from ui.ui import get_gradio


def main() -> None:
    # Mount Gradio UI at /ui on the same FastAPI server
    gr.mount_gradio_app(get_app(), get_gradio(predict_sentiment), path="/ui")
    api_run(predict_sentiment)


if __name__ == "__main__":
    main()
