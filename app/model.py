from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import torch.nn.functional as F 

MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def predict_batch(text: list[str]) -> list[dict]:  
    """
    Predict the sentiment for a list of input texts using a preloaded BERT model.

    Args:
        texts (list[str]): A list of input sentences or phrases to classify.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'label' (str): "positive" or "negative" sentiment prediction.
            - 'confidence' (float): Confidence score of the prediction (probability between 0 and 1).
    """
    inputs = tokenizer(text, return_tensors = 'pt', truncation = True, padding = True) 
    inputs = {k:v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(): 
        outputs = model(**inputs) 
        probs = F.softmax(outputs.logits, dim = 1) 
        labels = torch.argmax(probs, dim = 1).item() 
    
    results = []
    for i in range(len(text)): 
        label = 'positive' if labels[i] == 1 else 'negative'
        confidence = float(probs[i][labels[i]])
        results.append({'label': label, 'confidence': confidence})
    return results 




