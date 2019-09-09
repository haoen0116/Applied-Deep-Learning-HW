import torch


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        # TODO

        batch_size, samples_size = predicts.size()


        # 總共正確答案有 batch_size 個
        # 但predicct的結果，選的正確答案有幾個?

        self.n += batch_size
        predict_ans_index = predicts.argmax(1)  # 取出 predict 出來結果最大值之index
        for i in range(batch_size):
            # print(batch['labels'])
            # print(batch['labels'][i])
            # print(batch['labels'][i].argmax(0))

            if predict_ans_index[i] == batch['labels'][i].argmax(0):       # 若 predict 出來index的答案是0，則表示猜對了
                self.n_corrects += 1

        # print('predicts', predicts)
        # print('predicts.argmax()', predicts.argmax(1))
        # print('batch_labels', batch['labels'])



        # self.n += batch_size * samples_size
        # self.n_corrects += batch_size


        # This method will be called for each batch.
        # You need to
        # - increase self.n, which implies the total number of samples.
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch.

    def get_score(self):
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
