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
        # if not pd.isna(description):
        #     string_representation += f"The description is {description}. "

        return {
            'id': id,
            'text': string_representation
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

        return {
            'id': id,
            'text': string_representation
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
        if not pd.isna(release):
            string_representation += f"The release is {release}. "
        if not pd.isna(artist_name):
            string_representation += f"The artist is {artist_name}. "
        if year > 0:
            string_representation += f"The year is {year}. "
        if duration > 0:
            string_representation += f"The duration is {duration}. "

        return {
            'id': id,
            'text': string_representation
        }
