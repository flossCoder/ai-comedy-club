from transformers import pipeline, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import prepare_training_data
import pickle
import os
get_file_path = lambda : os.path.split(os.path.realpath(__file__))[0]
FILE_PATH_AUX = get_file_path()

from urllib.request import urlretrieve

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:32"

PKL_FILE_URL = "https://github.com/flossCoder/flossCoder_jokeBot_data/raw/main/joke_generator.pkl"

def build_pipeline():
    """
    This function generates the default pipeline for the dadsaysjokes model:
    https://huggingface.co/huggingtweets/dadsaysjokes
    CAUTION use this function only, if no further fine tuning shall be done.

    Returns
    -------
    generator : transformers.pipeline
        The default generator of the dadsaysjokes model.

    """
    generator = pipeline('text-generation', model='huggingtweets/dadsaysjokes')
    return generator

def build_model():
    """
    This function loads the gpt-2 based dadsaysjokes model:
    https://huggingface.co/huggingtweets/dadsaysjokes
    This model can be used for fine tuning.

    Returns
    -------
    model : transformers.GPT2PreTrainedModel
        The pre-trained dadsaysjokes model.

    """
    model = AutoModelWithLMHead.from_pretrained('huggingtweets/dadsaysjokes')
    return model

def build_trainer(model, train_TextDataset, val_TextDataset, number_of_epochs, tokenizer, filename = "joke_generator_model", wd = None, batch_size = 16, no_cuda = False, weight_decay = 0.01):
    """
    This function prepares the trainer for the given model on the jester training set.

    Parameters
    ----------
    model : transformers.GPT2PreTrainedModel
        This is the dadsaysjokes model, that we should train.
    train_TextDataset : TextDataset
        The TextDataset for the training set.
    val_TextDataset : TextDataset
        The TextDataset for the validation set.
    filename : string, optional
        The filename of the jester ratings dataset. The default is "joke_model".
    wd : string, optional
        The working directory of the model.
        The default is FILE_PATH_TRAIN_RATE_JOKE. Which is the path of this script.
        The default is indicated by None.
    number_of_epochs : int
        The number of training epochs.
    batch_size : int, optional
        The batch size, here we set per_device_train_batch_size = per_device_eval_batch_size = batch_size. The default is 64.
    no_cuda : bool, optional
        Use the cuda gpu, if possible. The default is False.
    weight_decay : float, optional
        The weight decay applied to all layers during the optimization.
        The default in the TrainingArguments objects of the transformers package
        is 0, which is a bit strange. The default is 0.01.

    Returns
    -------
    trainer : transformers.Trainer
        The trainer object after training. This object contains the best model.

    """
    training_args = TrainingArguments(
        output_dir = './training',
        num_train_epochs = number_of_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        weight_decay = weight_decay,
        no_cuda = no_cuda,
        load_best_model_at_end = True,
        evaluation_strategy = "epoch",
        save_strategy = "epoch"
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False
    )
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_TextDataset,
        eval_dataset = val_TextDataset,
        tokenizer = tokenizer,
        data_collator=data_collator
    )
    
    return trainer

def save_model(model, filename = "joke_generator", wd = None):
    """
    This function saves the model as pickle.

    Parameters
    ----------
    model : simpletransformers.classification.classification_model.ClassificationModel
        The given model which could be technically any python object.
        Nevertheless simpletransformers.classification.classification_model.ClassificationModel is the desired type.
    filename : string, optional
        The filename of the jester ratings dataset. The default is "joke_generator".
    wd : string, optional
        The working directory of the model.
        The default is FILE_PATH_TRAIN_RATE_JOKE. Which is the path of this script.
        The default is indicated by None.

    Returns
    -------
    None.

    """
    pickle.dump(model, open(os.path.join(FILE_PATH_AUX if wd is None else wd, "%s.pkl"%filename), "wb"))

def load_model(filename = "joke_generator", wd = None):
    """
    This function loads the pickled model.

    Parameters
    ----------
    filename : string, optional
        The filename of the jester ratings dataset. The default is "joke_generator".
    wd : string, optional
        The working directory of the model.
        The default is FILE_PATH_TRAIN_RATE_JOKE. Which is the path of this script.
        The default is indicated by None.

    Returns
    -------
    model : simpletransformers.classification.classification_model.ClassificationModel
        The given model which could be technically any python object.
        Nevertheless simpletransformers.classification.classification_model.ClassificationModel is the desired type.

    """
    filename = os.path.join(FILE_PATH_AUX if wd is None else wd, "%s.pkl"%filename)
    if not os.path.exists(filename):
        urlretrieve(PKL_FILE_URL, filename)
    model = pickle.load(open(filename, "rb"))
    return model

def train_joke_model(number_of_epochs):
    """
    The main function.

    Parameters
    ----------
    number_of_epochs : int
        The number of training epochs.

    Returns
    -------
    None.

    """
    model = build_model()
    tokenizer, train_TextDataset, test_TextDataset, val_TextDataset = prepare_training_data.prepare_training_data()
    trainer = build_trainer(model, train_TextDataset, val_TextDataset, number_of_epochs, tokenizer)
    trainer.train()
    trainer.save_model()
    save_model(model)

if __name__ == "__main__":
    train_joke_model(50)