from vqda import vqda
nlp = vqda(
    # Add yours models 
    we_model = './models/vqda_model/gensim_word_embedding/',   # Gensim
    qr_model = './models/vqda_model/t5_question_rewritting/',  # T5 model
)

question = "Tác phẩm nghệ thuật nào được coi là tác phẩm kinh điển của thế kỷ 20?"

# Random deletion
print(nlp.RD(question))

# Random swap
print(nlp.RS(question))

# Random insertion
print(nlp.RS(question))

# Synonym replacement
print(nlp.RS(question))

# nlp.QR(question)