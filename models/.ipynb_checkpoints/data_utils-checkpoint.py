import numpy as np

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
NUMBER = "[NUM]" # special token for all numbers

class REGDataset(object):
    """Class that iterates over REG Dataset

    __iter__ method yields a tuple (pre_context, pos_context, description, refex)
        pre_context: list of raw words
        pos_context: list of raw words
        description: list of raw words
        refex: list of raw words
        
    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = REGDataset(filename)
        for pre_context, pos_context, description, refex in data:
            pass
        ```

    """
    def __init__(self, dirname, processing_word=None, max_iter=None):
        """
        Args:
            dirname: path to the dir, in that dir must have 'entity.txt', 
                'refex.txt', 'pre_context.txt', 'pos_context.txt', 
                'description.txt'
            processing_words: (optional) function that takes a word as input, 
                convert word (str) in word_id (int)
            max_iter: (optional) max number of sentences to yield

        """
        self.dirname = dirname
        self.processing_word = processing_word
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        wiki_id_file = open(self.dirname+'/entity.txt', 'r', encoding='utf-8')
        refex_file   = open(self.dirname+'/refex.txt', 'r', encoding='utf-8')
        pre_context_file = open(self.dirname+'/pre_context.txt', 'r', encoding='utf-8')
        pos_context_file = open(self.dirname+'/pos_context.txt', 'r', encoding='utf-8')
        description_file = open(self.dirname+'/description.txt', 'r', encoding='utf-8')
        
        for _wiki_id, _refex, _pre_c, _pos_c, _des in zip(wiki_id_file, 
                                                          refex_file, 
                                                          pre_context_file, 
                                                          pos_context_file, 
                                                          description_file):
            _wiki_id, _refex, _pre_c, _pos_c, _des = _wiki_id.strip(), \
            _refex.strip(), _pre_c.strip(), _pos_c.strip(), _des.strip()
            
            if _wiki_id != '':
                niter += 1
                if self.max_iter is not None and niter > self.max_iter:
                    break
                
                # processing refex
                if self.processing_word is None:
                    refex = [_ for _ in _refex.split()]
                    pre_c = [_ for _ in _pre_c.split()]
                    pos_c = [_ for _ in _pos_c.split()]
                    des   = [_ for _ in _des.split()]
                else:
                    refex = [self.processing_word(i) for i in _refex.split()]
                    pre_c = [self.processing_word(i) for i in _pre_c.split()]
                    pos_c = [self.processing_word(i) for i in _pos_c.split()]
                    des   = [self.processing_word(i) for i in _des.split()]
                
                yield pre_c, pos_c, des, refex

        wiki_id_file.close()
        refex_file.close()
        pre_context_file.close()
        pos_context_file.close()
        description_file.close()
        
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
    
    
def get_processing_word(vocab_words=None, vocab_chars=None,
                        lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    for dataset in datasets:
        for pre_c, pos_c, des, refex in dataset:
            vocab_words.update(pre_c, pos_c, des, refex)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    word2id = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            word2id[word] = idx
    id2word = {v: k for k, v in word2id.items()}
    assert len(word2id) == len(id2word)
    return word2id, id2word


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)
    

def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for pre_c, pos_c, des, refex in dataset:
        for word in pos_c:
            vocab_char.update(word)
        for word in refex:
            vocab_char.update(word)

    return vocab_char