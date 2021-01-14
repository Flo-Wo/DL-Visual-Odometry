import logging, coloredlogs

from network_trainer import NetworkTrainer
from cnn.cnn_siamese_frames_flow import CnnSiamese
from data_loader import DatasetOptFlo1Frames


coloredlogs.install()
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    path_labels = "./data/raw/train_label.txt"
    network_save_file = "leakyReLU_Siamese"

    test_split_ratio = 0.8
    block_size = 3400

    dataLoader_params = {'batch_size': 64, 'shuffle': True}

    nwt = NetworkTrainer(20399, DatasetOptFlo1Frames, CnnSiamese)

    tr_tensor, eval_tensor = nwt.configure_data_loader(path_labels, test_split_ratio, block_size,
                                                       dataLoader_params, new_splitting=False)
    metadata = nwt.train_model(tr_tensor, eval_tensor, 3, 10, network_save_file)
