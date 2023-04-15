![GitHub](https://img.shields.io/github/license/sangcamap/vqda?color=s&style=for-the-badge)
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

question = "Shark Hưng đang giữ vị trí nào trong tập đoàn CENGROUP?"

# Random deletion
print(nlp.RD(question))
# >>> ['Shark đang giữ vị trí trong tập đoàn?', 
#      'Hưng đang nào trong tập đoàn CENGROUP?', 
#      'Shark đang giữ vị trí nào trong CENGROUP?']

# Random swap
print(nlp.RS(question))
# >>> ['Shark trong đang giữ vị trí nào CENGROUP tập đoàn Hưng?', 
#      'Tập đoàn Hưng đang giữ nào vị trí trong Shark CENGROUP?', 
#      'Shark Hưng đang nào CENGROUP giữ trong tập đoàn vị trí?']

# Random insertion
print(nlp.RI(question))
# >>> ['Shark Hưng đang shark giữ vị trí nào trong tập đoàn CENGROUP?', 
#      'Shark duy trì Hưng đang giữ vị trí nào trong tập đoàn CENGROUP?', 
#      'Shark League Two Hưng đang giữ vị trí nào trong tập đoàn CENGROUP?']

# Synonym replacement
print(nlp.RS(question))
# >>> ['Shark Phước đang giữ vị trí nào trong tập đoàn CENGROUP?', 
#      'Shark Hưng muốn giữ vị trí nào trong tập đoàn CENGROUP?', 
#      'Shark Hưng đang giữ vị trí nào cạnh tập đoàn CENGROUP?']

```

### Back translation

```python
from vqda import vqda

nlp = vqda()
question = "Shark Hưng đang giữ vị trí nào trong tập đoàn CENGROUP?"
print(nlp.BT(question))
# >>> ['Shark Hưng đang nắm giữ vị trí nào trong CENGROUP Group?']
```

### Question rewriting

```python
from vqda import vqda

nlp = vqda()
question = "Tại sao người ta dùng giun đất làm mồi câu cá?"
print(nlp.QP(question, n_aug = 5))
# >>> ['Vì sao người ta dùng giun đất làm mồi câu cá?',
# 'Tại sao người ta dùng giun đất làm mồi câu cá?',
# 'Lý do người ta dùng giun đất làm mồi câu cá?',
# 'Vì sao người ta sử dụng giun đất làm mồi câu cá?',
# 'Vì sao người ta dùng giun đất để câu cá?']
```


Example in [Notebook](https://github.com/sangcamap/vqda/blob/main/notebook.ipynb)
