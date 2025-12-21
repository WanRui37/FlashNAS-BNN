import torch


class Config:
    def __init__(self):
        self.model_type = "lstm"  # lstm or gru
        self.device = torch.device("cuda:0")
        # self.device = torch.device('cpu')

        """""" """""" """""" """
        训练参数
        learning_rate:学习率
        batch_size:批次大小
        epoches:训练轮数
        """ """""" """""" """"""
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.epoches = 64  # 2048

        """""" """""" """""" """
        通用参数
        embed_dim:词向量维数
        hidden_size:隐藏层大小
        dropout:丢弃概率
        mode = 0专门针对绝句
        mode = 1生成单个句子词数不一致
        convert_path:词汇表打印地址
        poem_path:生成诗打印地址
        loss_outpic_path:loss图片打印地址
        """ """""" """""" """"""
        self.layer_len_max = 32
        self.embed_dim = 20
        self.hidden_size = 256
        self.dropout = 0
        self.mode = 0
        self.np_best_accuracy_path = "./2-np_best_accuracy.npy"
        self.np_feature_single_vec_path = "./2-np_feature_single_vec.npy"
        self.np_feature_total_vec_path = "./2-np_feature_total_vec.npy"
        self.np_unique_channels_path = "./2-np_unique_channels.npy"
        self.loss_outpic_path = "./1-loss_lstm.png"
        self.search_output_path = "./1-output.txt"
