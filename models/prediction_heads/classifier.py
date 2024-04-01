import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_feats, output_feats):
        super(Classifier, self).__init__()
        mid_feats = (input_feats+output_feats)//2
        self.add_module("classifier", nn.Sequential(
                nn.Linear(in_features=input_feats, out_features=mid_feats),
                nn.Linear(in_features=mid_feats, out_features=output_feats),
                nn.Softmax()
            )
        )