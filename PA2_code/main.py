import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Encoder, Decoder
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(
        padded_sequences,
        (0, max(0, block_size - padded_sequences.shape[1])),
        "constant",
        0,
    )
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = 100 * total_correct / total_samples
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            _, loss, _ = decoderLMmodel(
                X, Y
            )  # your model should be computing the cross entropy loss
            losses.append(loss.item())
            # total_loss += loss.item()
            if len(losses) >= eval_iters:
                break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def main():
    torch.manual_seed(256)
    print("Loading data and creating tokenizer ...")
    texts = load_texts("speechesdataset")
    tokenizer = SimpleTokenizer(" ".join(texts))  # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(
        tokenizer, "speechesdataset/train_CLS.tsv"
    )
    train_CLS_loader = DataLoader(
        train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True
    )

    test_CLS_dataset = SpeechesClassificationDataset(
        tokenizer, "speechesdataset/test_CLS.tsv"
    )
    test_CLS_loader = DataLoader(
        test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True
    )

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, "r", encoding="utf-8") as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    inputfile_h = "speechesdataset/test_LM_hbush.txt"
    inputfile_o = "speechesdataset/test_LM_obama.txt"
    inputfile_w = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile_h, "r", encoding="utf-8") as f:
        lmtestText_h = f.read()
    test_LM_dataset_h = LanguageModelingDataset(tokenizer, lmtestText_h, block_size)
    test_LM_loader_h = DataLoader(
        test_LM_dataset_h, batch_size=batch_size, shuffle=True
    )
    with open(inputfile_o, "r", encoding="utf-8") as f:
        lmtestText_o = f.read()
    test_LM_dataset_o = LanguageModelingDataset(tokenizer, lmtestText_o, block_size)
    test_LM_loader_o = DataLoader(
        test_LM_dataset_o, batch_size=batch_size, shuffle=True
    )
    with open(inputfile_w, "r", encoding="utf-8") as f:
        lmtestText_w = f.read()
    test_LM_dataset_w = LanguageModelingDataset(tokenizer, lmtestText_w, block_size)
    test_LM_loader_w = DataLoader(
        test_LM_dataset_w, batch_size=batch_size, shuffle=True
    )

    # for the classification  task, you will train for a fixed number of epochs like this:
    CLS_model = Encoder(
        tokenizer.vocab_size,
        n_embd,
        n_head,
        n_layer,
        block_size,
        0,
        n_input,
        n_hidden,
        n_output,
    ).to(device)
    optimizer = torch.optim.AdamW(CLS_model.parameters(), lr=learning_rate)
    for epoch in range(epochs_CLS):
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss, attn_maps = CLS_model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_accuracy = compute_classifier_accuracy(CLS_model, train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(CLS_model, test_CLS_loader)
        print(
            f"Epoch: {epoch}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}"
        )
    util = Utilities(tokenizer, CLS_model)
    util.sanity_check(
        "Everywhere I look, I see people who have dedicated themselves to making sure that this day would come to pass: my friends from Congress, as I say, who worked so diligently with the best interest of all at heart, Democrats and Republicans; members of this administration—and I'm pleased to see so many top officials and members of my Cabinet here today who brought their caring and expertise to this fight; and then, the organizations—so many dedicated organizations for people with disabilities, who gave their time and their strength; and perhaps most of all, everyone out there and others—across the breadth of this nation are 43 million Americans with disabilities.",
        block_size,
    )

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    LM_model = Decoder(
        tokenizer.vocab_size,
        n_embd,
        n_head,
        n_layer,
        block_size,
        0,
        CLS_model.token_embedding_table,
        CLS_model.position_embedding_table,
    ).to(device)
    optimizer = torch.optim.AdamW(LM_model.parameters(), lr=learning_rate)
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        logits, loss, attn_maps = LM_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (i + 1) % eval_interval == 0:
            train_perplexity = compute_perplexity(LM_model, train_LM_loader, eval_iters)
            test_perplexity_h = compute_perplexity(
                LM_model, test_LM_loader_h, eval_iters
            )
            test_perplexity_o = compute_perplexity(
                LM_model, test_LM_loader_o, eval_iters
            )
            test_perplexity_w = compute_perplexity(
                LM_model, test_LM_loader_w, eval_iters
            )
            print(
                f"Iteration: {i}, Train Perplexity: {train_perplexity}, Test Perplexity hbush: {test_perplexity_h}, Test Perplexity obama: {test_perplexity_o}, Test Perplexity wbush: {test_perplexity_w}"
            )
    util = Utilities(tokenizer, LM_model)
    util.sanity_check(
        "Everywhere I look, I see people who have dedicated themselves to making sure that this day would come to pass: my friends from Congress, as I say, who worked so diligently with the best interest of all at heart, Democrats and Republicans; members of this administration—and I'm pleased to see so many top officials and members of my Cabinet here today who brought their caring and expertise to this fight; and then, the organizations—so many dedicated organizations for people with disabilities, who gave their time and their strength; and perhaps most of all, everyone out there and others—across the breadth of this nation are 43 million Americans with disabilities.",
        block_size,
    )


if __name__ == "__main__":
    main()
