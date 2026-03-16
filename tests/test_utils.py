import os
import tempfile
import yaml
from scripts.train import load_yaml_config

def test_load_yaml_config():
    # Create dummy config
    config_dict = {
        'model': {
            'input_window': 100,
            'forecast_horizon': 20
        },
        'dataset': {
            'type': 'monash',
            'monash_name': 'all',
            'num_series': 10
        }
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml', mode='w') as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        loaded_config = load_yaml_config(temp_path)
        assert loaded_config == config_dict
        assert loaded_config['model']['input_window'] == 100
        assert loaded_config['dataset']['type'] == 'monash'
    finally:
        os.remove(temp_path)
