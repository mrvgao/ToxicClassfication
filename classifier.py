import fasttext
import logging

logging.basicConfig(level=logging.INFO)

w2v_model_path = './cust_data/toxic_text_model.vec'
classifier = fasttext.supervised(
        './cust_data/train_corpus.txt', 
        './cust_data/toxic_clasifier_model', 
        dim=100, pretrained_vectors=w2v_model_path
)
