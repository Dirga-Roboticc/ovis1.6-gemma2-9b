import torch
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.float32,
                                             multimodal_max_length=8192,
                                             quantization_config=quantization_config,
                                             low_cpu_mem_usage=True,
                                             trust_remote_code=True,
                                             device_map="auto")
logger.info("Model loaded successfully")

# Load tokenizers
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()
logger.info("Tokenizers loaded successfully")

# Enter image path and prompt
image_path = input("Enter image path: ")
image = Image.open(image_path)
text = input("Enter prompt: ")
query = f'<image>\n{text}'

# Format conversation
prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
logger.info(f"Image preprocessed successfully. Pixel values shape: {pixel_values.shape}")

attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
input_ids = input_ids.unsqueeze(0).to(device=model.device, dtype=torch.long)
attention_mask = attention_mask.unsqueeze(0).to(device=model.device, dtype=torch.long)

# Ensure all tensors have the correct shape and dtype
pixel_values = pixel_values.to(device=model.device, dtype=model.dtype)
if pixel_values.dim() == 3:
    pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension if missing
elif pixel_values.dim() == 2:
    pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions if missing

# Ensure the first dimension (batch size) is 1
if pixel_values.shape[0] != 1:
    pixel_values = pixel_values.unsqueeze(0)

input_ids = input_ids.to(device=model.device, dtype=torch.long)
attention_mask = attention_mask.to(device=model.device, dtype=torch.long)

logger.info(f"Pixel values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
logger.info(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
logger.info(f"Attention mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")

# Generate output
with torch.inference_mode():
    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True
    )
    output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
    output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f'Output:\n{output}')
