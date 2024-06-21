import yaml
import pickle
import os
import datetime


def save_model(model,config):
    """Save the model to the models directory with a timestamp."""
    models_dir = 'models'

    #Create a unique model_name based on model's name and date at which it was ran
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    my_model_name = config['model_name']
    model_name = config['model']
    model_filename = f"{my_model_name}_{model_name}__{current_time}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved to {model_path}")


def save_config(config):
    """Save the configs of the model that were run"""
    configs_dir = 'config'

    #Create a unique name of config 
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    my_model_name = config['model_name']
    config_filename = f"{my_model_name}_config__{current_time}.yaml"
    config_path = os.path.join(configs_dir,config_path)

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    print(f"Model saved to {model_path}")

    
