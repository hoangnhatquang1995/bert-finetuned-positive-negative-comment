import argparse
from ui.ui import ui_run
from prediction import predict_sentiment

def main() -> None:
	
	ui_run(predict_sentiment)

if __name__ == "__main__":
	main()
