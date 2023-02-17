[![PyPI - License](https://img.shields.io/hexpm/l/plug)](https://github.com/sangcamap/vqda/blob/main/LICENSE)
[![visitors](https://visitor-badge.glitch.me/badge?page_id=vqda.count_visitors)](https://visitor-badge.glitch.me)

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
    we_model = './models/vqda_model/gensim_word_embedding/',   # Gensim
    qr_model = './models/vqda_model/t5_question_rewritting/',  # T5 model
)

question = "Tác phẩm nghệ thuật nào được coi là tác phẩm kinh điển của thế kỷ 20?"

# Random deletion
print(nlp.RD(question))
# >>> ['Nào coi là tác phẩm kinh điển của 20?', 'Tác phẩm nghệ thuật nào được coi là tác phẩm kinh điển của thế kỷ 20?', 'Tác phẩm nghệ thuật nào được tác phẩm kinh điển của thế kỷ 20?']

# Random swap
print(nlp.RS(question))
# >>> ['Tác phẩm thế kỷ nào được coi nghệ thuật tác phẩm kinh điển của là 20?', 'Coi nghệ thuật nào được Tác phẩm là tác phẩm kinh điển thế kỷ của 20?', 'Tác phẩm nghệ thuật 20 được coi là kinh điển tác phẩm của thế kỷ nào?']

# Random insertion
print(nlp.RS(question))
# >>> ['Tác phẩm nghệ thuật nào tác phẩm coi là 20 kinh điển của thế kỷ được?', 'Được nghệ thuật nào thế kỷ coi là tác phẩm kinh điển của Tác phẩm 20?', 'Tác phẩm nghệ thuật nào thế kỷ coi là được kinh điển của tác phẩm 20?']

# Synonym replacement
print(nlp.RS(question))
# >>> ['Tác phẩm nghệ thuật nào được thế kỷ là Tác phẩm kinh điển của coi 20?', 'Tác phẩm nghệ thuật tác phẩm được nào là coi kinh điển của thế kỷ 20?', 'Tác phẩm của nào thế kỷ coi là tác phẩm kinh điển nghệ thuật được 20?']

```
