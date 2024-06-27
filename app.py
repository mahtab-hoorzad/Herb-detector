from fastcore.all import *
from fastai.vision.all import *
import gradio as gr

learn = load_learner('herb.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(height=512, width=512),
    outputs=gr.Label(num_top_classes=3)
).launch()
