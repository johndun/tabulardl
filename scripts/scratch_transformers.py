"""
https://huggingface.co/docs/transformers/autoclass_tutorial
https://huggingface.co/docs/transformers/preprocessing
https://towardsdatascience.com/feature-extraction-with-bert-for-text-classification-533dde44dc2f
"""
from transformers import AutoTokenizer
from transformers import AutoModel


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))

model = AutoModel.from_pretrained("distilbert-base-uncased")

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(batch)

output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
