import torch
import numpy as np
from transformers import BertTokenizer, BertModel, GPT2Model, GPT2Tokenizer
import scraper


def encode(model, tokenizer, texts):
    encoding = tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
                                           return_tensors='pt', add_special_tokens=True)

    with torch.no_grad():
        outputs = model(encoding['input_ids'], attention_mask=encoding['attention_mask'])
        word_embeddings = outputs.last_hidden_state.mean(dim=1)

    return word_embeddings.detach().numpy()


def vectorize(model, tokenizer):
    data = scraper.get_data()

    features_vect = []
    features = []
    labels_vect = []
    labels = []

    for entry in data:
        x = entry[1]
        y = entry[0]
        if x == "" or y == "" or len(x) != 16 or len(y) != 4:
            continue

        combined = x + y

        curr = encode(model, tokenizer, combined)
        features.append(np.array(x))
        labels.append(np.array(y))
        features_vect.append(curr[:16])
        labels_vect.append(curr[16:])

    return np.stack(features_vect), np.stack(features), np.stack(labels_vect), np.stack(labels)


if __name__ == "__main__":
    tk = BertTokenizer.from_pretrained('bert-base-uncased')
    md = BertModel.from_pretrained('bert-base-uncased')
    a, b, c, d = vectorize(md, tk)
    # a is N x 16 x 768
    # b is N x 16
    # c is N x 4 x 768
    # d is N x 4

    np.save('bert_ft_vect.npy', a)
    np.save('bert_ft.npy', b)
    np.save('bert_lb_vect.npy', c)
    np.save('bert_lb.npy', d)
