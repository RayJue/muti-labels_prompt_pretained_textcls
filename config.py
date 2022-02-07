from ast import mod
import torch





class Config(object):
    def __init__(self,model_name = ''):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##地址类
        self.saved_dir = '/home/rayjue/python_project/fidh_kgqa/fish_kgqa-master/saved/intent_model' # 模型保存路径
        self.pretrained_dir = '/home/rayjue/share/public_model/bert-base-chinese' #预训练模型地址
        self.data_dir = '/home/rayjue/python_project/fidh_kgqa/fish_kgqa-master/data/intent' #数据地址
        self.config_path = '/home/rayjue/share/public_model/bert-base-chinese/config.json'
        self.model_path = '/home/rayjue/share/public_model/bert-base-chinese/pytorch_model.bin'
        self.vocab_path = '/home/rayjue/share/public_model/bert-base-chinese/vocab.txt'
        

        ##标签类
        self.labels = []
        self.label2id = [i for i in self.labels]

        ##训练参数
        self.epoches = 5
        self.batch_size = 16
        self.lr = 0.002 # 初始学习率
        self.dropout = 0.5
        self.pad_size = 128 # 短填长切
        self.num_classes = len(self.labels)
        self.emb_size = 768 
        # self.hid_size = 256
        self.mid = True
        self.weight_decay = 0.0001
        
        self.model_name = model_name

        #多模型设置
        if self.model_name == 'cnn':
            self.filter_sizes = (2, 3, 4) # 卷积核尺寸
            self.num_filters = 256 # 卷积核数目 channels
        elif self.model_name == 'lstm':
            self.hid_size = 128  # LSTM隐藏层
            self.num_layers = 2  # lSTM层数
        elif self.model_name == 's2s':
            pass
        else:
            pass






