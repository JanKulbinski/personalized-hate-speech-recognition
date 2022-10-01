import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME_PATH = "../data/"