# Imports used during the implementation not shown in slides
from http.cookiejar import LoadError

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import random
from rich import print
from rich.progress import Progress
from collections import Counter
        
from sklearn.model_selection import train_test_split

import urllib.request       
import os      


class LanguageModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(block_size * embedding_dim, hidden_dim)
        self.hidden_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)           # (batch, block_size, embedding_dim)
        emb = emb.view(emb.size(0), -1)  # (batch, block_size * embedding_dim)
        h = F.tanh(self.fc1(emb))
        h = F.tanh(self.hidden_1(h))
        
        return self.fc2(h)            # logits (batch, vocab_size)

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ff(x))
        return x


class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, block_size: int):
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks        = nn.Sequential(*[AttentionBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.fc            = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.blocks(x)
        return self.fc(x[:, -1, :])  # predict from last token only


def download_data() -> None:
    '''  Load the names and store them into the names.txt!  '''
    # check if names are loaded
    if 'names.txt' not in os.listdir():                                                                                          
        urllib.request.urlretrieve(                                                                                                            
            "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",                                                            
            "names.txt"                                                                                                                        
    )
        print(f'[yellow]INFO:[/yellow] names.txt has been loaded!')
    else:
        print(f'[yellow]INFO:[/yellow] names.txt has been found!')
    
    
def print_names(list_of_names: list[str], color = 'blue') -> None:
    print('Names:')
    for name in list_of_names:
        print(f'    [{color}]{name}[/{color}]')


def look_and_get_data(verbose: bool = False) -> list[str]:
    with open('names.txt', 'r') as f: 
        data = f.read().splitlines()
        if verbose:
            print(f'[yellow]INFO:[/yellow] The dataset consist of [blue]{len(data)}[/blue] names.')
            print_names(data[:10])
        return data
    raise LoadError('Could not Load the from names.txt!')


def preprocess(data: list[str]) -> list[str]:
    new = sorted(list(set(data)))
    return new


def plot_name_length_histogram(names, prefix=''):
    plt.figure(figsize=(10, 8))
    name_lengths = list(map(len, names))
    plt.hist(name_lengths, bins=max(name_lengths))
    # plt.yscale('log')
    plt.xticks(range(min(name_lengths), max(name_lengths)))
    plt.xlabel('Length of name')
    plt.ylabel('Frequency')
    plt.title(prefix+'Histogram of name lengths')

    mean_length = np.mean(name_lengths)
    std_dev_length = np.std(name_lengths)

    # Add mean and standard deviation lines
    plt.axvline(mean_length, color='r', linestyle='dashed', linewidth=1, label= f'Mean: {mean_length:.2f}')
    plt.axvline(mean_length + std_dev_length, color='g', linestyle='dashed', linewidth=1, label=f' + 1 SD: {mean_length + std_dev_length:.2f}')
    plt.axvline(mean_length - std_dev_length, color='b', linestyle='dashed', linewidth=1, label=f' - 1 SD: {mean_length - std_dev_length:.2f}')
    
    # move ticks positions under their bars
    position, lbls = plt.xticks()
    plt.xticks(position+.5, lbls)

    # Add a legend
    plt.legend()
    plt.show()
    

def plot_char_frequency(names, prefix = '') -> None:
    # Count the frequency of each character
    char_counts = Counter(''.join(names))
    
    # Extract the keys and values from the Counter
    sorted_items = sorted(char_counts.items())
    
    # Extract the labels and values
    labels, counts = zip(*sorted_items)
    colors = plt.cm.hsv([i/len(labels) for i in range(len(labels))])
    
    plt.figure(figsize=(6, 3))
    plt.bar(labels, counts, color=colors)
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.title(prefix+'Frequency of characters')
    plt.show()
    

def map_chars_to_ints(data: list[str]) -> tuple[dict[str, int], dict[int, str]]: 
    chars = sorted(list(set(''.join(data))))
    str_to_int = {s:i+1 for i, s in enumerate(chars)}  # mapping chars to ints
    str_to_int['.'] = 0
    int_to_str = {i:s for s, i in str_to_int.items()}  # mappint ints to chars
    
    return str_to_int, int_to_str
    

def build_dataset(words: List[str], block_size: int, verbose: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dataset for training a language model.

    Args:
        words (List[str]): List of strings containing names.
        block_size (int): Size of the context window.
        verbose (Optional[bool], optional): If True, print verbose output. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing input data (X) and target data (Y) tensors.
    """
    X, Y = [], []
    
    str_to_int, int_to_str = map_chars_to_ints(words)

    for w in words:
        if verbose:
            print(f"\n{w}")
        
        context = [0] * block_size  # Initialize a context window of zeros
        for ch in w + '.':
            ix = str_to_int[ch]           # Convert character to integer index using stoi
            X.append(context)       # Append the current context window to X
            Y.append(ix)            # Append the index of the current character to Y
            if verbose:
                print(''.join(int_to_str[i] for i in context), '--->', int_to_str[ix])
            
            context = context[1:] + [ix] # Update the context window by cropping the first element and appending the new index.
    # Convert X and Y lists to PyTorch tensors
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def train(model: nn.Module, X_train: torch.Tensor, Y_train: torch.Tensor,
          X_val: torch.Tensor, Y_val: torch.Tensor,
          lr: float = 0.01, epochs: int = 10, batch_size: int = 64) -> nn.Module:

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train),
        batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in pbar:
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_loss = F.cross_entropy(model(X_train), Y_train).item()
        val_loss   = F.cross_entropy(model(X_val), Y_val).item()
    print(f"Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
    return model


def generate_names(model: nn.Module, int_to_str: dict, block_size: int, n: int = 10, max_length: int = 20) -> None:
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            context = [0] * block_size
            name = []
            for _ in range(max_length):
                x = torch.tensor([context])
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                if ix == 0:
                    break
                context = context[1:] + [ix]
                name.append(int_to_str[ix])
            print(f"[green]{''.join(name)}[/green]")


def main():
    # process params
    analyse_processed_data = False
    verbose = True
    VOCAB_SIZE = 27  # + '.'
    
    # Hyperparameter
    block_size = 8
    seed = 1
    
    
    # ====== PREPROCESSING =====
    with Progress() as progress:
        task = progress.add_task('Download Data...', total = 3)
        download_data()
        
        data = look_and_get_data(verbose = verbose)
        
        progress.update(task, advance = 1, description = 'Clear Data...')
        clear_data = preprocess(data)
        
        # analyse statistics
        if analyse_processed_data:
            plot_name_length_histogram(clear_data)
            plot_char_frequency(data)
        
        
        # maps
        progress.update(task, advance = 1, description = 'Build Dataset...')
        str_to_int, int_to_str = map_chars_to_ints(clear_data)
        X, Y = build_dataset(clear_data, block_size=block_size, verbose = False)
        
        #if analyse_processed_data:
        print(X.shape, X.dtype, Y.shape, Y.dtype)


        data_train, data_test = train_test_split(data, train_size=0.7, shuffle=True, random_state=seed)
        # data_val, data_test   = train_test_split(data_test, train_size=0.5, random_state=seed)

        X_train, Y_train = build_dataset(data_train, block_size=block_size)
        X_val, Y_val     = build_dataset(data_test, block_size=block_size)
       # X_test, Y_test   = build_dataset(data_test, block_size=block_size)
        
        progress.update(task, advance = 1, description = 'Finished Preprocessing!')
        
    
    model = TransformerLanguageModel(vocab_size=VOCAB_SIZE, embed_dim=64, num_heads=4, num_layers=2, block_size=block_size)

    trained_model = train(model, X_train, Y_train, X_val, Y_val, epochs = 5)

    generate_names(trained_model, int_to_str, block_size, n=10)
    

if __name__ == '__main__':
    main()