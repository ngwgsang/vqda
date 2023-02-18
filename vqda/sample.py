from vqda import vqda
nlp = vqda(
    # Add yours models 
    we_model = './models/vqda_model/gensim_word_embedding/vda.size5000.bin',   # Gensim
    qr_model = './models/vqda_model/t5_vietnamese_qr',  # T5 model
    vi2en_model = './models/others/vinai-translate-vi2en',
    en2vi_model = './models/others/vinai-translate-en2vi',
)

question = "Shark Hưng đang giữ vị trí nào trong tập đoàn CENGROUP?"

# Random deletion
print(nlp.RD(question))

# Random swap
print(nlp.RS(question))

# Random insertion
print(nlp.RI(question))

# Synonym replacement
print(nlp.SR(question))

# Back translation
print(nlp.BT(question))

# Question rewritting
print(nlp.QR(question))

