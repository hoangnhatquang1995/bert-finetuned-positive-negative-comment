import argparse
from ui.ui import ui_run
from api.api import api_run
from prediction import predict_sentiment,get_model

def main() -> None:
	# ui_run(predict_sentiment)
	api_run(predict_sentiment)

if __name__ == "__main__":
	main()
