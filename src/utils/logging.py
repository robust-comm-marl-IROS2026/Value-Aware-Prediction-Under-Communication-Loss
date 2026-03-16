from collections import defaultdict
import logging
import numpy as np
import os
import csv
from datetime import datetime

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])
        self.csv_file = None
        self.csv_writer = None

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def setup_file_logging(self, log_dir, unique_token):
        """Setup CSV file logging for metrics"""
        os.makedirs(log_dir, exist_ok=True)
        
        # Create CSV file
        csv_file_path = os.path.join(log_dir, f"{unique_token}.csv")
        self.csv_file = open(csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        self.csv_writer.writerow(['timestamp', 'timestep', 'episode', 'metric', 'value'])
        self.csv_file.flush()
        
        self.console_logger.info(f"CSV logging initialized: {csv_file_path}")

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]
            
            self._run_obj.log_scalar(key, value, t)
        
        # Write to CSV
        if self.csv_writer is not None:
            try:
                # Get episode number for this timestep
                episode = self.stats.get("episode", [(t, 0)])[-1][1] if "episode" in self.stats else 0
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Handle numpy types
                if hasattr(value, 'item'):
                    value = value.item()
                
                self.csv_writer.writerow([timestamp, t, episode, key, value])
                self.csv_file.flush()
            except Exception as e:
                pass

    def log_model(self, name, filepath, to_sacred=True):
        if self.use_sacred and to_sacred:
            self._run_obj.add_artifact(filepath, filepath)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)
    
    def close(self):
        """Close CSV file properly"""
        if self.csv_file is not None:
            self.csv_file.close()


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger