import gradio as gr
from prediction import calculate_predicted_price


with gr.Blocks() as iface:
    gr.Markdown(
        """
        # House Price Prediction
        Enter the details of the house to predict its price.
        """
    )
    with gr.Row():
        with gr.Column():
            basic_info = [
                gr.Number(label="Number of bedrooms"),
                gr.Number(label="Number of bathrooms"),
                gr.Number(label="Number of stories"),
                gr.Number(label="Number of parking slots"),
                gr.Number(label="Area size (in square feet)"),
            ]
        with gr.Column():
            amenities = [
                gr.Radio(label="Main Road (1 for yes, 0 for no)", choices=["0", "1"]),
                gr.Radio(label="Guest Room (1 for yes, 0 for no)", choices=["0", "1"]),
                gr.Radio(label="Basement (1 for yes, 0 for no)", choices=["0", "1"]),
                gr.Radio(label="Hot Water Heating (1 for yes, 0 for no)", choices=["0", "1"]),
                gr.Radio(label="Air Conditioning (1 for yes, 0 for no)", choices=["0", "1"]),
                gr.Radio(label="Preferred Area (1 for yes, 0 for no)", choices=["0", "1"]),
            ]
        with gr.Column():
            preferences = [
                gr.Dropdown(label="Furnishing Status", choices=["furnished", "semi-furnished", "unfurnished"])
            ]
    output = gr.Textbox(label="Predicted Price")
    btn = gr.Button("Predict")
    btn.click(calculate_predicted_price, inputs=[*basic_info, *amenities, *preferences], outputs=output)

if __name__ == "__main__":
    iface.launch()
