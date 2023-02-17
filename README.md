![GitHub](https://img.shields.io/github/license/sangcamap/vqda?style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/sangcamap/vqda?style=for-the-badge)
![GitHub Repo stars](https://img.shields.io/github/stars/sangcamap/vqda?style=for-the-badge)

# vqda
vqda is to provide data augmentation methods for Vietnamese questions.

## Install

```
pip install git+https://github.com/sangcamap/vqda.git
```

## Quick start

### Augment

```python
from vqda import vqda

nlp = vqda(
    # Add yours models 
    we_model = './models/vqda_model/gensim_word_embedding/vda.size5000.bin',   # Gensim
    qr_model = './models/vqda_model/t5_question_rewritting/',  # T5 model
)

question = "Tác phẩm nghệ thuật nào được coi là tác phẩm kinh điển của thế kỷ 20?"

# Random deletion
print(nlp.RD(question))
# >>> ['Tác phẩm nào được tác phẩm kinh điển của thế kỷ 20?', 
#      'Tác phẩm nào được coi là tác phẩm của 20?', 
#      'Tác phẩm được coi là tác phẩm kinh điển thế kỷ 20?']

# Random swap
print(nlp.RS(question))
# >>> ['Tác phẩm được nào nghệ thuật coi thế kỷ tác phẩm kinh điển của là 20?', 
#      '20 nghệ thuật nào coi được là tác phẩm kinh điển của thế kỷ Tác phẩm?', 
#      'Tác phẩm nghệ thuật nào được của là tác phẩm kinh điển coi 20 thế kỷ?']

# Random insertion
print(nlp.RI(question))
# >>> ['Tổ chức nào đang Tổ Chức cố gắng giải quyết vấn đề xã hội tại Đông Nam Á?', 
#      'Tổ chức nỗ lực nào đang cố gắng giải quyết vấn đề xã hội tại Đông Nam Á?', 
#      'Tổ chức khắc phục nào đang cố gắng giải quyết vấn đề xã hội tại Đông Nam Á?']

# Synonym replacement
print(nlp.RS(question))
# >>> ['Tổ chức nào đang cố gắng khắc phục vấn đề xã hội tại Đông Nam Á?', 
#      'Tổ chức thay thế nào đang cố gắng giải quyết vấn đề xã hội tại Đông Nam Á?', 
#      'Tổ chức nào đang cố gắng giải quyết vấn đề xã hội Tại Đông Nam Á?']

```

### Question rewriting

```python
print(nlp.QR(question))
# >>> ['Tổ chức nào đang cố gắng giải quyết vấn đề xã hội tại Đông Nam Á?', 
#      'Tổ chức nào đang cố gắng giải quyết vấn đề xã hội ở Đông Nam Á?', 
#      'Tổ chức nào đang cố gắng giải quyết vấn đề xã hội?']
```