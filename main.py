from dataset import make_batches,train_examples,val_examples,tokenizers
from position import PositionalEmbedding

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for (pt, en), en_labels in train_batches.take(1):
    embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
    embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)

    pt_emb = embed_pt(pt)
    en_emb = embed_en(en)
print(en_emb._keras_mask)