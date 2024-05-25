import os
import pandas as pd
import pytest
import pickle
from click.testing import CliRunner
from main import train, cli
from numpy import random


# paths
TEST = 'test_small.csv'
TRAIN = 'train_small.csv'
MODEL = 'model.pkl'


@pytest.fixture
def cleanup_model():
    yield
    if os.path.exists(MODEL):
        os.remove(MODEL)


# train
@pytest.mark.usefixtures('cleanup_model')
def test_valid_train():

    runner = CliRunner()
    result = runner.invoke(cli, ['train', '--data', TRAIN, 
                                 '--model', MODEL, '--split', '0.2'])

    assert os.path.isfile(MODEL)
    assert result.exit_code == 0


def test_non_existent_train():

    runner = CliRunner()
    result = runner.invoke(cli, ['train', '--data', 'non_existent_train.csv',
                                 '--model', MODEL, '--split', '0.2'])
    
    assert result.exit_code != 0


# prediction
@pytest.mark.usefixtures('cleanup_model')
def test_valid_prediction():

    prev_dir_files = set(os.listdir(os.getcwd()))

    runner = CliRunner()
    runner.invoke(cli, ['train', '--data', TRAIN, '--model', MODEL, '--split', '0.2'])
    result = runner.invoke(cli, ['predict', '--model', MODEL,
                                 '--data', 'I recommend this company to everyone'])

    prev_dir_files.add(MODEL)
    cur_dir_files = set(os.listdir(os.getcwd()))

    assert cur_dir_files == prev_dir_files
    assert result.exit_code == 0
    

# split
@pytest.mark.usefixtures('cleanup_model')
def test_split_size():

    df = pd.read_csv(TRAIN)
    split = 0.2

    runner = CliRunner()
    result = runner.invoke(train, ['--data', TRAIN,
                                         '--model', MODEL, '--split', str(split)])
    
    assert int(result.output.split()[5]) == (len(df) * split)


@pytest.mark.usefixtures('cleanup_model')
def test_split_shuffling():
    
    # фиксируем разные random_seed
    random_seeds = [random.randint(1, 100) for _ in range(10)]
    unique_preds = set()

    for seed in random_seeds:
        runner = CliRunner()
        result = runner.invoke(train, ['--data', TRAIN, '--model', MODEL,
                                       '--split', 0.2, '--seed', str(seed)])
        unique_preds.add(result.output.split()[1])

    assert len(unique_preds) > 1
