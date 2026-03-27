import gradio as gr

def get_gradio(predict_fn) -> gr.Interface:
    app = gr.Interface(
        fn=predict_fn,
        inputs=gr.Textbox(lines=3, placeholder="Nhập bình luận..."),
        outputs=gr.Label(num_top_classes=2),
        title="Sentiment (PhoBERT fine-tuned)",
        description="Demo deploy nhanh trên Colab: POSITIVE/NEGATIVE",
        examples=[
            ["Quả táo này rất ngon"] ,
            ["Quả táo này rất dở"] ,
        ],
     )
    return app


def ui_run(predict_fn) -> None:
    app = get_gradio(predict_fn)
    app.launch(share=True, debug=False)
