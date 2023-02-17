from vqda import vqda
nlp = vqda(
    # Add yours models 
    we_model = './models/vqda_model/gensim_word_embedding/vda.size5000.bin',   # Gensim
    qr_model = 'sangcamap/t5_vietnamese_qr',  # T5 model
)

question = "Tổ chức nào đang cố gắng giải quyết vấn đề xã hội tại Đông Nam Á?"

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

