import re
import random
from underthesea import word_tokenize, pos_tag
from gensim.models.word2vec import Word2Vec
import warnings
import copy
import random
from random import shuffle
from tqdm import tqdm
from glob import glob
from simplet5 import SimpleT5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class vqda:
    
    def __init__(
                self, 
                word_embedding_model = './vietnamese_word_embedding_5000', 
                question_paraphrasing_model = './vietnamese_question_paraphrasing_ViT5_base',
                vi2en_model = 'vinai/vinai-translate-vi2en',
                en2vi_model = 'vinai/vinai-translate-en2vi',
                stop_words = [],
                special_chars = '',
                gpu = False):

        warnings.filterwarnings('ignore')
        try:
          self.word_embedding_model = Word2Vec.load( glob(word_embedding_model + "/*.bin")[0])
        except:
          raise Exception(f"Can't use {word_embedding_model}")
        
        try:
          self.question_paraphrasing_model = SimpleT5()
          self.question_paraphrasing_model.load_model("t5", question_paraphrasing_model, use_gpu = gpu)
        except:
          raise Exception(f"Can't use {question_paraphrasing_model}")
        
        try:
          self.tokenizer_vi2en = AutoTokenizer.from_pretrained(vi2en_model, src_lang="vi_VN")
          self.tokenizer_en2vi = AutoTokenizer.from_pretrained(en2vi_model, src_lang="en_XX")
          self.model_vi2en = AutoModelForSeq2SeqLM.from_pretrained(vi2en_model)
          self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(en2vi_model)
        except:
          raise Exception(f"Can't use {vi2en_model} and {en2vi_model}")

        self.stop_words = stop_words
        self.special_chars = special_chars
    

    def get_only_chars(self, sentence):
        clean_sentence = ""
        for char in sentence:
            if char in 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ1234567890/' + self.special_chars:
                clean_sentence += char
            else:
                clean_sentence += ' '

        clean_sentence = re.sub(' +',' ',clean_sentence)
        if clean_sentence[0] == ' ':
            clean_sentence = clean_sentence[1:]

        return clean_sentence


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

    def get_synonyms(self, word, top_n = 4):
        try:
              list_synonym = self.word_embedding_model.wv.most_similar(word, topn = top_n )
              list_synonym = [token[0] for token in list_synonym] 
        except:
              list_synonym = []                    

        if not list_synonym:
            return [word]
        else:
            return list_synonym


    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            
            if num_replaced >= n:
                break

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

    def random_deletion(self, words, alpha):
        if len(words) == 1:
          return words

        new_words = []
        for word in words:
          r = random.uniform(0, 1)
          if r > alpha:
            new_words.append(word)

        if len(new_words) == 0:
          rand_int = random.randint(0, len(words)-1)
          return [words[rand_int]]

        return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(new_words)-1)
                counter += 1
                if counter > 3:
                    return new_words

        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words


########################################################################
# Random addition
# Randomly add n words into the sentence
########################################################################

    def random_addition(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
      synonyms = []
      counter = 0
      while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = self.get_synonyms(random_word)
        counter += 1
        if counter >= 10:
          return

      random_synonym = synonyms[0]
      random_idx = random.randint(0, len(new_words)-1)
      new_words.insert(random_idx, random_synonym)
  

########################################################################
# Auto complete question 
# Make first token is uppercase
# Add ? at the end

    def complete_question(self, question):
      question = question.strip()
      question = question[0].upper() + question[1:]
      if question[-1] != "?":
          question += "?"
      return question


########################################################################
# Main data augmentation function
#
# * eda 4
# * RD
# * RS
# * RI
# * SR
# * BT
#
########################################################################

    def eda_4(
              self, 
              sentence, 
              n_aug = 5, 
              alpha_rd = 0.2, 
              alpha_rs = 0.2, 
              alpha_ri = 0.2, 
              alpha_sr = 0.2,
              auto_complete_question = True
            ):
        
        sentence = self.get_only_chars(sentence)
        words =   word_tokenize(sentence) # thay vì split từng từ thì dùng underthesea để cắt từ theo nghĩa
        words = [word for word in words if word != '']
        num_words = len(words)
        
        augmented_sentences = []
        num_new_per_technique = int(n_aug/4)+1
        n_sr = max(1, int(alpha_sr*num_words))
        n_ri = max(1, int(alpha_ri*num_words))
        n_rs = max(1, int(alpha_rs*num_words))

        #sr
        for _ in range(num_new_per_technique):
          a_words = self.synonym_replacement(words, n_sr)
          augmented_sentences.append(' '.join(a_words))

        #ri
        for _ in range(num_new_per_technique):
          a_words = self.random_addition(words, n_ri)
          augmented_sentences.append(' '.join(a_words))

        #rs
        for _ in range(num_new_per_technique):
          a_words = self.random_swap(words, n_rs)
          augmented_sentences.append(' '.join(a_words))

        #rd
        for _ in range(num_new_per_technique):
          a_words = self.random_deletion(words, alpha_rd)
          augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [self.get_only_chars(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)

       
        if n_aug >= 1:
          augmented_sentences = augmented_sentences[:n_aug]
        else:
          keep_prob = n_aug / len(augmented_sentences)
          augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
        
        if (auto_complete_question):
          return [self.complete_question(sentence) for sentence in augmented_sentences]
        else:
          return augmented_sentences

    def SR(
            self, 
            sentence, 
            n_aug = 3, 
            alpha = 0.2,
            auto_complete_question = True
            ):
        sentence = self.get_only_chars(sentence)
        words = word_tokenize(sentence)
        num_words = len(words)
        augmented_sentences = []
        n_sr = max(1, int(alpha * num_words))
        for _ in range(n_aug):
          a_words = self.synonym_replacement(words, n_sr)
          augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [self.get_only_chars(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)
        # augmented_sentences.append(sentence)

        if (auto_complete_question):
          return [self.complete_question(sentence) for sentence in augmented_sentences]
        else:
          return augmented_sentences

    def RI(
            self, 
            sentence, 
            n_aug = 3, 
            alpha = 0.2,
            auto_complete_question = True
            ):
        sentence = self.get_only_chars(sentence)
        words = word_tokenize(sentence)
        num_words = len(words)
        augmented_sentences = []
        n_ri = max(1, int(alpha*num_words))

        for _ in range(n_aug):
          a_words = self.random_addition(words, n_ri)
          augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [self.get_only_chars(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)
        # augmented_sentences.append(sentence)

        if (auto_complete_question):
          return [self.complete_question(sentence) for sentence in augmented_sentences]
        else:
          return augmented_sentences

    def RS(
            self, 
            sentence, 
            n_aug = 3, 
            alpha = 0.3,
            auto_complete_question = True
            ):
        sentence = self.get_only_chars(sentence)
        words = word_tokenize(sentence)
        num_words = len(words)
        augmented_sentences = []
        n_rs = max(1, int(alpha*num_words))
        for _ in range(n_aug):
          a_words = self.random_swap(words, n_rs)
          augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [self.get_only_chars(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)
        # augmented_sentences.append(sentence)

        if (auto_complete_question):
          return [self.complete_question(sentence) for sentence in augmented_sentences]
        else:
          return augmented_sentences

    def RD(
            self, 
            sentence, 
            n_aug = 3, 
            alpha = 0.2,
            auto_complete_question = True
            ):
        sentence = self.get_only_chars(sentence)
        words = word_tokenize(sentence)
        words = [word for word in words if word != '']
        augmented_sentences = []

        for _ in range(n_aug):
          a_words = self.random_deletion(words, alpha)
          augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [self.get_only_chars(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)
        # augmented_sentences.append(sentence)

        if (auto_complete_question):
          return [self.complete_question(sentence) for sentence in augmented_sentences]
        else:
          return augmented_sentences

    def QP(self, question, n_aug = 3, prefix = "question paraphrasing"):
        augmented_sentences = self.question_paraphrasing_model.predict(f"{prefix}: {question}", num_return_sequences= n_aug , num_beams= n_aug )
        return augmented_sentences
    
    def BT(self, question, translator = 'vinai'):
        if translator == 'vinai':
          def translate_vi2en(vi_text: str) -> str:
              input_ids = self.tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
              output_ids = self.model_vi2en.generate(
                  input_ids,
                  do_sample=True,
                  top_k=100,
                  top_p=0.8,
                  decoder_start_token_id=self.tokenizer_vi2en.lang_code_to_id["en_XX"],
                  num_return_sequences=1,
              )
              en_text = self.tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
              en_text = " ".join(en_text)
              return en_text

          def translate_en2vi(en_text: str) -> str:
              input_ids = self.tokenizer_en2vi(en_text, return_tensors="pt").input_ids
              output_ids = self.model_en2vi.generate(
                  input_ids,
                  do_sample=True,
                  top_k=100,
                  top_p=0.8,
                  decoder_start_token_id=self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
                  num_return_sequences=1,
              )
              vi_text = self.tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
              vi_text = " ".join(vi_text)
              return vi_text
        
        question =  translate_vi2en(question)
        question =  translate_en2vi(question)
        return [question]


