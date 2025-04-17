from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class BaseDataset(Dataset):

    def __init__(self, file_path, tokenizer):
        self.df = pd.read_csv(file_path, encoding='latin1')
        self.tokenizer = tokenizer

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
        # if not pd.isna(description):
        #     string_representation += f"The description is {description}. "

        input_ids = self.tokenizer.encode(string_representation, return_tensors='pt', truncation=True).squeeze()

        return {
            'id': id,
            'input_ids': input_ids
        }

class GoogleDataset(BaseDataset):
    def __getitem__(self, idx):
        id=self.df.iloc[idx]['id']
        title = self.df.iloc[idx]['name']
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
        # if not pd.isna(description):
        #     string_representation += f"The description is {description}. "

        input_ids = self.tokenizer.encode(string_representation, return_tensors='pt', truncation=True).squeeze()

        return {
            'id': id,
            'input_ids': input_ids
        }