import gradio as gr
from transformers import pipeline

# Load the image classification model
image_classification = pipeline("image-classification", model="ombhojane/healthyPlantsModel")

# Function to predict labels for the given image URL
def predict_labels(image_url):
    result = image_classification(image_url)
    return result[0]

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_labels,
    inputs=gr.Image(type="url", label="Input Image URL"),
    outputs=gr.Textbox(label="Predicted Label"),
    live=True
)

# Launch the Gradio interface
iface.launch()
