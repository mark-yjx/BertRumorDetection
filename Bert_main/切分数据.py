import pandas as pd

train_data_path = './CHEF/train.tsv'
dev_data_path = './CHEF/dev.tsv'
test_data_path = './CHEF/test.tsv'
train_df = pd.read_csv(train_data_path, sep='\t', header=None)
dev_df = pd.read_csv(dev_data_path, sep='\t', header=None)
test_df = pd.read_csv(test_data_path, sep='\t', header=None)


train_len = train_df.shape[0]
dev_len = dev_df.shape[0]
test_len = test_df.shape[0]

all_data = pd.concat([train_df, dev_df, test_df])

all_data = all_data.sample(frac=1)

