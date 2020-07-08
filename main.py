from train import Trainer
from seqGAN.config import getConfig
config = getConfig("config.ini")
maxLen = config["max_length"]


trainer = Trainer(
    config["batch_size"],
    config["max_length"],
    config["d_e"],
    config["d_h"],
    config["c_e"],
    config["d_dropout"],
    config["path_pos"],
    config["path_neg"],
    config["g_lr"],
    config["d_lr"],
    config["n_sample"],
    config["generate_samples"]
)
trainer.preTrain(g_epoch=config["g_pre_epochs"],
                 d_epoch=config["d_pre_epochs"],
                 g_pre_path="data/save/generator_pre.hdf5",
                 d_pre_path="data/save/discriminator_pre.hdf5",
                 g_lr=config["g_pre_lr"],
                 d_lr = config["d_pre_lr"])
trainer.train(
    steps=10,
    g_steps=20,
    d_steps =5,
    g_weights_path=config["g_weights_path"],
    d_weights_path=config["d_weights_path"]

)
trainer.generate_txt()