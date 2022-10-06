import configparser
import os

CONFIG_FPATH = './config.ini'
class Config:
    
    def __init__(self):
        self.config = configparser.ConfigParser()
        if not os.path.exists(CONFIG_FPATH):
            self.config['OMERO'] = {"host": "omero-imaging.uni-muenster.de", 
                                    "user": "", 
                                    "port":"4064"}
            self.config['N2V'] =   {"n_epoch": "100", 
                                    "batch_size":"32",
                                    "patch_size":"64",
                                    "train_fraction":"0.9",
                                    "steps_per_epoch":"200",
                                    "neighborhood_radius":"2"}
            self.config["PATH"] = {"models_folder":os.path.join(os.path.expanduser("~"), "models_n2v"),
                                   "model_name": "my_model"}
            self.config["DISPLAY"] = {"max_side":"2048"}
            self.write_config()
        else:
            # Read File
            self.config.read(CONFIG_FPATH)
            
    def save_config(self):
        self.write_config()
        
    def write_config(self):
        self.config.write(open(CONFIG_FPATH, 'w'))
        
    def __getitem__(self,name):
        return self.config[name]