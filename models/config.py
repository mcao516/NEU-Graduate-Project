import os

from datetime import datetime
from .general_utils import get_logger
from .data_utils import load_vocab, get_processing_word, get_trimmed_glove_vectors
from .data_utils import START_DECODING, STOP_DECODING, UNKNOWN_TOKEN


class Config():
    def __init__(self, load=True, operation='train'):
        """Initialize hyperparameters and load vocabs
        
        Args:
            load_embeddings: (bool) if True, load embeddings into np array, else None

        """
        self.dir_output = "results/{}/{:%Y%m%d_%H%M%S}/".format(operation, 
            datetime.now())
        self.dir_model  = self.dir_output + "model/"
        self.path_log   = self.dir_output + "log.txt"
        self.path_summary = self.dir_output + "summary"

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.word2id, self.id2word = load_vocab(self.filename_words)
        self.char2id, self.id2char = load_vocab(self.filename_chars)
        self.pronoun2id, _ = load_vocab(self.filename_pronouns)

        self.npronouns  = len(self.pronoun2id)
        self.nwords     = len(self.word2id)
        self.nchars     = len(self.char2id)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.word2id,
                self.char2id, lowercase=True, chars=self.use_chars)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)
        
        self.start_id = self.word2id[START_DECODING]
        self.stop_id = self.word2id[STOP_DECODING]
        self.unk_id = self.word2id[UNKNOWN_TOKEN]

    multi_gpu = True
    print_samples = False

    # embeddings
    dim_word = 100
    dim_char = 50

    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_dev = "data/webnlg/WebNLG_dev.pickle"
    filename_test = "data/webnlg/WebNLG_test.pickle"
    filename_train = "data/webnlg/WebNLG_train.pickle"
    
    # split_type = 'separated'
    # filename_dev = "data/webnlg/re-split-{0}/WebNLG_{0}_dev.pickle".format(split_type)
    # filename_test = "data/webnlg/re-split-{0}/WebNLG_{0}_test.pickle".format(split_type)
    # filename_train = "data/webnlg/re-split-{0}/WebNLG_{0}_train.pickle".format(split_type)
    filename_sample = "data/webnlg/samples.pickle"

    max_iter = None # if not None, max number of examples in Dataset
    max_enc_context = 80
    max_enc_description = 80

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_chars = "data/chars.txt"
    filename_pronouns = "data/pronouns.txt"
    
    use_chars = True
    nepochs   = 100
    nepoch_no_imprv = 100
    char_emb_type = 'rnn'
    
    # model hyper parameters
    beam_size = 4
    max_dec_steps = 10
    min_dec_steps = 3
    beam_search = False
    reverse_pos_context = False
    use_context_words = False
    drop_out = 0.20 # 0.5
    shuffle_dataset = True
    scheduler = 'cosine'
    
    lr = 0.001
    lr_decay = 0.90
    l2_reg = 1e-6
    batch_size = 64
    batch_size_eva = 1
    grad_clip = 5
    
    hidden_size = 100
    hidden_size_char = 50
 
    def log_info(self):
        self.logger.info("Embedding file: {}".format(self.filename_glove))
        self.logger.info("Training file: {}".format(self.filename_train))
        self.logger.info("Word Embedding Size: {}".format(self.dim_word))
        if self.use_chars:
            self.logger.info("Char Embedding Size: {}".format(self.dim_char))
        self.logger.info("Scheduler Type: {}".format(self.scheduler))
        self.logger.info("Learning Rate: {:.2}".format(self.lr))
        self.logger.info("Learning Rate Decay: {:.2}".format(self.lr_decay))
        self.logger.info("L2 Regularization Rate: {:.2}".format(self.l2_reg))
        self.logger.info("Dropout Rate: {:.2}".format(self.drop_out))
        self.logger.info("Batch size: {}".format(self.batch_size))
        self.logger.info("Hidden layer size: {}".format(self.hidden_size))
        self.logger.info("Hidden char size: {}".format(self.hidden_size_char))
        self.logger.info("Earyly Stop: {}".format(self.nepoch_no_imprv))