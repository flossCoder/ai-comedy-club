import random
import os
import importlib.util
import torch
import sys
spec_joke_bot = importlib.util.spec_from_file_location("joke_bot", "./bots/flossCoder_jokeBot/joke_bot.py")
module_joke_bot = importlib.util.module_from_spec(spec_joke_bot)
sys.modules["module.name"] = module_joke_bot
spec_joke_bot.loader.exec_module(module_joke_bot)

spec_custom_pyjokes = importlib.util.spec_from_file_location("joke_bot", "./bots/flossCoder_jokeBot/custom_pyjokes.py")
module_custom_pyjokes = importlib.util.module_from_spec(spec_custom_pyjokes)
sys.modules["module.name"] = module_custom_pyjokes
spec_custom_pyjokes.loader.exec_module(module_custom_pyjokes)


get_file_path = lambda : os.path.split(os.path.realpath(__file__))[0]
sys.path.insert(0, get_file_path())

from train_joke_model import load_model
from prepare_training_data import prepare_tokenizer
from transformers import pipeline

class Bot(module_joke_bot.Bot):
    """
    This bot is an extension to the flossCoder jokeBot.
    It uses a generative model to obtain new jokes. It utilizes the "old" joke
    generation for the proposals of the model.
    """
    name = 'flossCoder jokeBot2'
    def __init__(self, filename_joke_model = "joke_model", wd_joke_model = None, filename_joke_generator = "joke_generator", wd_joke_generator = None):
        """
        Set up our flossCoder jokeBot2.

        Parameters
        ----------
        filename_joke_model : string, optional
            The filename of the jester ratings dataset for the super class.
            The default is "joke_model".
        wd_joke_model : string, optional
            The working directory of the model for the super class
            The default is FILE_PATH_TRAIN_RATE_JOKE. Which is the path of this script.
            The default is indicated by None. The default is None.
        filename_joke_generator : string, optional
            The filename of the jok generator model. The default is "joke_generator".
        wd_joke_generator : string, optional
            The working directory of the generator model. The default is None.

        Returns
        -------
        None.

        """
        super().__init__(filename_joke_model, wd_joke_model)
        self.textGeneration_pipeline = self.setup_pipeline(filename_joke_generator, wd_joke_generator)
        self.custom_pyjoke_rater = self.Constant_Rater()
    
    def tell_joke(self, category = None, min_size = 3):
        """
        This function tells jokes. And can be used by the main program and the test_bot.
        The whole interaction with the user will be done seperately.

        Parameters
        ----------
        category : String, optional
            The category states the category, from which the joke shall be taken.
            In case None is given, the category will be drawn randomly.
            The default is None.
        min_size : int, optional
            The minimum size of the joke part. In case the joke is smaller than
            the min_size, the joke is returned. The default is 3.

        Returns
        -------
        joke : string
            The joke, that I want to tell.

        """
        joke_internal = self.tell_super_joke(category)
        joke_part = self.get_joke_part(joke_internal, min_size)
        joke = self.get_joke(joke_part)
        return joke
    
    def get_joke(self, joke_part):
        """
        This function generates a joke based on the given joke_part and the pipeline.

        Parameters
        ----------
        joke_part : string
            The joke part, that I want to tell.

        Returns
        -------
        joke : string
            The joke, that I want to tell.

        """
        joke = self.textGeneration_pipeline(joke_part)[0]['generated_text']
        return joke
    
    def get_joke_part(self, joke, min_size = 3):
        """
        This function obtains a part of a joke.

        Parameters
        ----------
        joke : string
            The joke, that I want to tell.
        min_size : int, optional
            The minimum size of the joke part. In case the joke is smaller than
            the min_size, the joke is returned. The default is 3.

        Returns
        -------
        joke_part : string
            The joke part, that I want to tell.

        """
        joke_split = joke.split(" ")
        if len(joke_split) > min_size:
            return " ".join(joke_split[:random.randint(3,len(joke_split))])
        else:
            return joke
            
    
    def tell_super_joke(self, category = None):
        """
        This function tells a joke analogiously to the super().tell_joke.
        The difference is, that the rating is done by the self.generator_model,
        this avoids expensive execution of the super rating function.

        Parameters
        ----------
        category : String, optional
            The category states the category, from which the joke shall be taken.
            In case None is given, the category will be drawn randomly.
            The default is None.

        Returns
        -------
        joke : string
            The joke, that I want to tell.

        """
        if category is None:
            category = random.choices(self.joke_categories)[0]
        joke = module_custom_pyjokes.get_joke(self.custom_pyjoke_rater, category)
        return joke
    
    def setup_pipeline(self, filename_joke_generator, wd_joke_generator):
        """
        This function initializes the pipeline for joke generation based on the fine-tuned model.

        Parameters
        ----------
        filename_joke_generator : string, optional
            The filename of the jok generator model. The default is "joke_generator".
        wd_joke_generator : string, optional
            The working directory of the generator model. The default is None.

        Returns
        -------
        textGeneration_pipeline : transformers.pipelines.text_generation.TextGenerationPipeline
            The actual pipeline generating jokes based on an input proposal.

        """
        model = load_model(filename_joke_generator, wd_joke_generator)
        tokenizer = prepare_tokenizer()
        textGeneration_pipeline = pipeline('text-generation', model = model, tokenizer = tokenizer, device = torch.cuda.current_device())
        return textGeneration_pipeline
    
    class Constant_Rater():
        def __init__(self, rate_constant = 1):
            """
            Set up the constant rater, this is required to prevent the super tell_joke from rating the joke proposals.

            Parameters
            ----------
            rate_constant : double, optional
                The constant returned while rating a joke. The default is 1.

            Returns
            -------
            None.

            """
            self.rate_constant = rate_constant
        
        def rate_joke(self, joke):
            """
            

            Parameters
            ----------
            joke : string
                The given joke, that should be rated.

            Returns
            -------
            rating : float
                The constant rating of the bot.

            """
            return self.rate_constant
    
if __name__ == "__main__":
    bot = Bot()
    bot.interactive_joking()