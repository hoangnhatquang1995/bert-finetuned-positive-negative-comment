import argparse
from ui.ui import ui_run,get_gradio
from api.api import api_run,get_app
from prediction import predict_sentiment,get_model
import gradio as gr

def main() -> None:
	# ui_run(predict_sentiment)
	gr.mount_gradio_app(get_app(),get_gradio(predict_sentiment), path="/ui")
	api_run(predict_sentiment)

if __name__ == "__main__":
	main()
