from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class BaseDataset(Dataset):

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, encoding='latin1')

    def __len__(self):
        return len(self.df)

class AmazonDataset(BaseDataset):
    def __getitem__(self, idx):
        id = self.df.iloc[idx]['id']
        title = self.df.iloc[idx]['title']
        description = self.df.iloc[idx]['description']
        manufacturer = self.df.iloc[idx]['manufacturer']
        price = self.df.iloc[idx]['price']

        string_representation = ""
        if not pd.isna(title):
            string_representation += f"The product is {title}. "
        if not pd.isna(manufacturer):
            string_representation += f"It is manufactured by {manufacturer}. "
        if price != 0:
            string_representation += f"The price is {price}. "
        if not pd.isna(description):
            string_representation += f"The description is {description}. "

        return {
            'id': id,
            'text': string_representation
        }

class GoogleDataset(BaseDataset):
    def __getitem__(self, idx):
        id=self.df.iloc[idx]['id']
        title = self.df.iloc[idx]['title']
        description = self.df.iloc[idx]['description']
        manufacturer = self.df.iloc[idx]['manufacturer']
        price = self.df.iloc[idx]['price']
        string_representation = ""
        if title is not None:
            string_representation += f"The product is {title}. "
        if manufacturer is not None:
            string_representation += f"It is manufactured by {manufacturer}. "
        if price is not 0:
            string_representation += f"The price is {price}. "
        if description is not None:
            string_representation += f"The description is {description}. "

        return {
            'id': id,
            'text': string_representation
        }


if __name__ == '__main__':
    # get current directoru

    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, '../data/amazon_google/Amazon.csv')
    amazon_dataset = AmazonDataset(data_path)

    # create a dataloader of batch size = 128
    amazon_dataloader = DataLoader(amazon_dataset, batch_size=128)
    for i, data in enumerate(amazon_dataloader):
        print(data)
        break