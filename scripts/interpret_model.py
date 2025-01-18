import shap
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

MODEL_DIR = "../models/fine_tuned/"
TEXT_EXAMPLES = [
    "አዲስ አበባ ውስጥ የተካሄደ እንቅስቃሴ ሰልፍ እንደሆነ ተነግሯል።",
    "የተኩስ ግምት በህዝብ ውስጥ ስለ መኖሩ ተነገረ።"
]

def interpret_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

    def model_wrapper(inputs):
        tokenized_inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
        outputs = model(**tokenized_inputs)
        return np.array(outputs.logits.detach().numpy())

    explainer = shap.Explainer(model_wrapper, TEXT_EXAMPLES)
    shap_values = explainer(TEXT_EXAMPLES)

    shap.summary_plot(shap_values)

if __name__ == "__main__":
    interpret_model()
