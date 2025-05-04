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
        manufacturer = self.df.iloc[idx]['manufacturer']
        price = self.df.iloc[idx]['price']

        string_representation = ""
        if not pd.isna(title):
            string_representation += f"The product is {title}. "
        if not pd.isna(manufacturer):
            string_representation += f"It is manufactured by {manufacturer}. "
        if price != 0:
            string_representation += f"The price is {price}. "

        input_ids = self.tokenizer.encode(string_representation, return_tensors='pt', truncation=True).squeeze()

        return {
            'id': id,
            'input_ids': input_ids
        }

class GoogleDataset(BaseDataset):
    def __getitem__(self, idx):
        id=self.df.iloc[idx]['id']
        title = self.df.iloc[idx]['name']
        manufacturer = self.df.iloc[idx]['manufacturer']
        price = self.df.iloc[idx]['price']

        string_representation = ""
        if not pd.isna(title):
            string_representation += f"The product is {title}. "
        if not pd.isna(manufacturer):
            string_representation += f"It is manufactured by {manufacturer}. "
        if price != 0:
            string_representation += f"The price is {price}. "

        input_ids = self.tokenizer.encode(string_representation, return_tensors='pt', truncation=True).squeeze()

        return {
            'id': id,
            'input_ids': input_ids
        }

class SongsDataset(BaseDataset):
    def __getitem__(self, idx):
        id = self.df.iloc[idx]['id']
        title = self.df.iloc[idx]['title']
        release = self.df.iloc[idx]['release']
        artist_name = self.df.iloc[idx]['artist_name']
        duration = self.df.iloc[idx]['duration']
        artist_familiarity = self.df.iloc[idx]['artist_familiarity']
        artist_hotness = self.df.iloc[idx]['artist_hotttnesss']
        year = self.df.iloc[idx]['year']

        string_representation = ""
        if not pd.isna(title):
            string_representation += f"The song title is {title}. "
        else:
            string_representation += f"The song title is unknown. "
        if not pd.isna(release):
            string_representation += f"The release is {release}. "
        else:
            string_representation += f"The release is unknown. "
        if not pd.isna(artist_name):
            string_representation += f"The artist is {artist_name}. "
        else:
            string_representation += f"The artist is unknown. "
        if duration > 0:
            string_representation += f"The duration is {duration}. "
        else:
            string_representation += f"The duration is unknown. "
        if artist_familiarity > 0:
            string_representation += f"The artist familiarity is {artist_familiarity}. "
        else:
            string_representation += f"The artist familiarity is unknown. "
        if artist_hotness > 0:
            string_representation += f"The artist hotness is {artist_hotness}. "
        else:
            string_representation += f"The artist hotness is unknown. "

        input_ids = self.tokenizer.encode(string_representation, return_tensors='pt', truncation=True).squeeze()

        return {
            'id': id,
            'input_ids': input_ids
        }