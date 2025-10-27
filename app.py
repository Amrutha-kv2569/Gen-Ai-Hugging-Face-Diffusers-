import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

st.set_page_config(page_title="Sketch to Image (CPU Demo)", layout="wide")

st.title("🎨 Sketch to Real Image Generator (CPU-Compatible)")
st.write("Upload a black-and-white sketch, and this model generates a realistic image (slow but works on CPU).")

# Load model only once
@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.to("cpu")
    return pipe

pipe = load_model()

uploaded_file = st.file_uploader("Upload your sketch (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    sketch = Image.open(uploaded_file).convert("RGB")
    st.image(sketch, caption="Uploaded Sketch", width=300)

    if st.button("Generate Realistic Image"):
        with st.spinner("Generating... (this will take 1–2 minutes on CPU)"):
            # Convert sketch image to prompt text
            prompt = "a realistic image of the object drawn in the sketch"
            result = pipe(prompt, num_inference_steps=20)
            gen_image = result.images[0]

            st.image(gen_image, caption="Generated Realistic Image", width=400)
            buf = io.BytesIO()
            gen_image.save(buf, format="PNG")
            st.download_button("Download Image", data=buf.getvalue(), file_name="generated.png", mime="image/png")
