#! /usr/bin/python3
import pandas as pd


def write_filtered_df(df, file):
    filtered_df = df[df.gold_label != '-']
    filtered_df = filtered_df[['sentence1', 'sentence2', 'gold_label']]
    filtered_df.columns = ['premise', 'hypothesis', 'label']
    filtered_df.to_csv('.data/snli-1.0-formatted-{}.csv'.format(file), header=True, index=False, sep='\t')


def filter_data():
    base_path = '.data/snli_1.0/{}'
    train_file = base_path.format('snli_1.0_train.jsonl')
    val_file = base_path.format('snli_1.0_dev.jsonl')
    test_file = base_path.format('snli_1.0_test.jsonl')

    train_df = pd.read_json(train_file, lines=True)
    write_filtered_df(train_df, 'train')
    val_df = pd.read_json(val_file, lines=True)
    write_filtered_df(val_df, 'val')
    test_df = pd.read_json(test_file, lines=True)
    write_filtered_df(test_df, 'test')


if __name__ == '__main__':
    filter_data()
