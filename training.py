
from model import *
from config_model import *


class Train(object):
    def __init__(self):
        with open('./data/data_preparation/dict.pkl', 'rb') as f:
            map_dict = pickle.load(f)
        self.net = Model(map_dict)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="mean")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def train(self):
        self.net.train()
        # with open('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/dict.pkl', 'rb') as f:

        train_manager = BatchManager(batch_size=2, name='train_small')
        for epoch in range(1, epoches):
            print("epoch {}".format(epoch))
            for id, batch in enumerate(train_manager.iter_batch(shuffle=True)):
                # print(batch)
                print("batch {}".format(id))
                words, labels, bounds, flags, radicals, pinyins = get_fea(batch)
                logit = self.net.forward(words, bounds, flags, radicals, pinyins)
                logit = logit.view(logit.size(0) * logit.size(1), logit.size(2))
                labels = labels.view(labels.size(0) * labels.size(1))
                # print(logit.size())
                # print(labels.size())
                loss = self.loss_function(logit, labels)
                print(loss.item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


if __name__ == '__main__':
    re = Train()
    re.train()
    # data / data_preparation / dict.pkl