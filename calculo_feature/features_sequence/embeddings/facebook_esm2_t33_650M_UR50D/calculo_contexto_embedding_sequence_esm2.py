from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")