import torch
import numpy as np


class NTCentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTCentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, num_class, batch_index):
        # num_class shape: (1)
        # batch_index shape: (N)
        diag = np.eye(num_class) # [M, M]
        mask_pos = torch.from_numpy((diag)).to(self.device)
        mask_pos = mask_pos.index_select(0, batch_index) # [N, M]
        mask_neg = (1 - mask_pos).type(torch.bool)
        mask_pos = mask_pos.type(torch.bool)
        return mask_pos, mask_neg

    @staticmethod
    def _dot_simililarity(x, y):
        # x shape: (N, 1, C)
        # y shape: (1, C, N)
        # v shape: (N, N)
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, rep, fp, target):
        # rep shape: (N, C)
        # fp shape: (M, C)
        # target shape: (N)

        # compute similarity matrix
        similarity_rep2fp = self.similarity_function(rep, fp) # [N, M]
        similarity_fp2rep = self.similarity_function(fp, rep).T # [N, M]

        # get masks from the batch labels
        mask_pos, mask_neg = self._get_correlated_mask(fp.shape[0], target)

        # filter out the scores from the positive samples
        l_pos = similarity_rep2fp[mask_pos] # [N]
        r_pos = similarity_fp2rep[mask_pos] # [N]
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1) # [2N, 1]

        # filter out the scores from the negative samples
        l_neg = similarity_rep2fp[mask_neg].view(self.batch_size, -1) # [N, M-1]
        r_neg = similarity_fp2rep[mask_neg].view(self.batch_size, -1) # [N, M-1]
        negatives = torch.cat([l_neg, r_neg]) # [2N, M-1]

        logits = torch.cat((positives, negatives), dim=1) # [2N, M]
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long() # [2N]
        loss = self.criterion(logits, labels) # [1]

        return loss / (2 * self.batch_size)