import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embedding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    emb_matrix = np.random.uniform(-.08, .08, size=(len(vocab), emb_size))

    with open(emb_file, 'r') as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            if word in vocab:
                vec = np.array(tokens[1:], dtype=np.float32)
                # ensure that embed size matches
                if len(vec) == emb_size:
                    idx = vocab[word]
                    emb_matrix[idx] = vec
                else:
                    print(f'Word {word} has embed size mismatch: {len(vec)}, expected {emb_size}')

    return emb_matrix


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        # Embedding
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)

        # FeedForward
        layers = []
        for i in range(0, self.args.hid_layer):
            layer = nn.Linear(self.args.emb_size, self.args.hid_size)
            layers.append(layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.args.hid_drop))
        self.hidden_layers = nn.Sequential(*layers)

        # Output
        self.output_layer = nn.Linear(self.args.hid_size, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        # if pre-trained embeddings provided, this will be overwritten so this is fine
        nn.init.uniform_(self.embedding.weight, -0.08, 0.08)

        # using kaiming uniform because of the hidden layers using ReLU activations
        for layer in self.hidden_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb_matrix = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        emb_tensor = torch.tensor(emb_matrix, torch.float32)
        self.embedding.weight.data.copy_(emb_tensor)

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        embed = self.embedding(x)
        # implement embedding dropout here maybe?
        avg_embed = embed.mean(dim=1)
        ff_output = self.hidden_layers(avg_embed)
        scores = self.output_layer(ff_output)
        return scores
