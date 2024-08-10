import sys
sys.path.append('../flossCoder_jokeBot')
import os
get_file_path = lambda : os.path.split(os.path.realpath(__file__))[0]
FILE_PATH_AUX = get_file_path()

from transformers import AutoTokenizer, TextDataset

from aux import load_jester_items, train_test_val_split

def save_training_data_to_file(jester_train_set, jester_test_set, jester_val_set, wd = None, jester_train_filename = "jester_train.csv", jester_test_filename = "jester_test.csv", jester_val_filename = "jester_val.csv"):
    """
    This function saves the training data to file.

    Parameters
    ----------
    jester_train_set : pandas DataFrame
        The train set of the jester set dataset.
    jester_test_set : pandas DataFrame
        The test set of the jester set dataset.
    jester_val_set : pandas DataFrame
        The validation set of the jester set dataset.
    wd : string, optional
        The working directory of the jester dataset.
        The default is None. In this case the directory of the python script is used.
    jester_train_filename : string, optional
        The filename of the jester train set. The default is "jester_train.csv".
    jester_test_filename : string, optional
        The filename of the jester test set. The default is "jester_test.csv".
    jester_val_filename : string, optional
        The filename of the jester val set. The default is "jester_val.csv".

    Returns
    -------
    jester_train_fn : string
        The absolute filepath of the jester train set.
    jester_test_fn : string
        The absolute filepath of the jester test set.
    jester_val_fn : string
        The absolute filepath of the jester val set.

    """
    jester_train_fn = provide_filename(wd, jester_train_filename)
    jester_test_fn = provide_filename(wd, jester_test_filename)
    jester_val_fn = provide_filename(wd, jester_val_filename)
    jester_train_set.to_csv(jester_train_fn)
    jester_test_set.to_csv(jester_test_fn)
    jester_val_set.to_csv(jester_val_fn)
    return jester_train_fn, jester_test_fn, jester_val_fn

def provide_filename(wd, filename):
    """
    Provide the absolute path of the file.

    Parameters
    ----------
    wd : string
        The working directory of the jester dataset.
        In case of None the directory of the python script is used.
    filename : string
        The filename.

    Returns
    -------
    string
        The absolute path of the file.

    """
    return os.path.join(FILE_PATH_AUX, filename) if wd is None else os.path.join(wd, filename)

def provide_wd(wd = None):
    """
    Provide the correct absolute wd string.

    Parameters
    ----------
    wd : string, optional
        The working directory of the jester dataset.
        The default is None. In this case the directory of the python script is used.

    Returns
    -------
    string
        The absolute wd string.

    """
    return FILE_PATH_AUX if wd is None else wd

def provide_TextDatasets(tokenizer, wd = None, jester_train_filename = "jester_train.csv", jester_test_filename = "jester_test.csv", jester_val_filename = "jester_val.csv", buffer_size = 1000, seed = 42):#GEHT NICHT besser: from datasets import Dataset
    """
    This function creates the IterableDatasetDicts, that are required to stream the data from hard drive.

    Parameters
    ----------
    tokenizer : transformers.models.TOKENIZER
        The input tokenizer.
    train_fn : string
        The absolute filepath of the train set.
    test_fn : string
        The absolute filepath of the test set.
    val_fn : string
        The absolute filepath of the val set.

    Returns
    -------
    train_TextDataset : TextDataset
        The TextDataset for the training set.
    test_TextDataset : TextDataset
        The TextDataset for the test set.
    val_TextDataset : TextDataset
        The TextDataset for the validation set.
    
    """
    wd = provide_wd(wd)
    train_TextDataset = TextDataset(
        tokenizer=tokenizer,
        file_path=jester_train_filename if wd is None else os.path.join(wd, jester_train_filename),
        block_size=128)
    test_TextDataset = TextDataset(
        tokenizer=tokenizer,
        file_path=jester_train_filename if wd is None else os.path.join(wd, jester_test_filename),
        block_size=128)
    val_TextDataset = TextDataset(
        tokenizer=tokenizer,
        file_path=jester_train_filename if wd is None else os.path.join(wd, jester_val_filename),
        block_size=128)
    
    return train_TextDataset, test_TextDataset, val_TextDataset

def prepare_tokenizer():
    """
    Obtain the tokenizer.

    Returns
    -------
    tokenizer : transformers.models.TOKENIZER
        The tokenizer.

    """
    tokenizer = AutoTokenizer.from_pretrained("huggingtweets/dadsaysjokes")
    return tokenizer

def prepare_training_data():
    """
    The main function.

    Returns
    -------
    tokenizer : transformers.models.TOKENIZER
        The tokenizer.
    train_TextDataset : TextDataset
        The TextDataset for the training set.
    test_TextDataset : TextDataset
        The TextDataset for the test set.
    val_TextDataset : TextDataset
        The TextDataset for the validation set.

    """
    tokenizer = prepare_tokenizer()
    jester_items = load_jester_items()
    jester_train_set, jester_test_set, jester_val_set = train_test_val_split(jester_items)
    [jester_train_fn, jester_test_fn, jester_val_fn] = save_training_data_to_file(jester_train_set, jester_test_set, jester_val_set)
    train_TextDataset, test_TextDataset, val_TextDataset = provide_TextDatasets(tokenizer)
    return tokenizer, train_TextDataset, test_TextDataset, val_TextDataset

if __name__ == "__main__":
    tokenizer, train_TextDataset, test_TextDataset, val_TextDataset = prepare_training_data()
