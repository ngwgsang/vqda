from vqda import vqda
nlp = vqda(
    # Add yours models 
    we_model = './models/vqda_model/gensim_word_embedding/vda.size5000.bin',   # Gensim
    qr_model = './models/vqda_model/t5_question_rewritting/',  # T5 model
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

# # nlp.QR(question)

# from gensim.models.word2vec import Word2Vec
# model = Word2Vec.load("./models/vqda_model/gensim_word_embedding/vda.size5000.bin")
# print(model.wv.most_similar("quê hương"))
