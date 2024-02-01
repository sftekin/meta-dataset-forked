import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from fusionshot import train_fusion


def roll_data(input_data, window_size=1000, stride=100):
    for i in range(0, len(input_data) - window_size, stride):
        yield input_data[i:i + window_size]


def get_base_pred(in_data, model_names, n_way):
    y = in_data[:, -1]
    base_acc = []
    for i in range(0, len(model_names) * n_way, n_way):
        base_acc.append(np.mean(in_data[:, i:i + n_way].argmax(axis=1) == y) * 100)
    return base_acc


def run():
    all_labels = np.load("all_labels.npy")
    all_logits = np.load("all_logits.npy")
    all_data = np.concatenate([all_logits, all_labels[:, None]], axis=1)
    model_names = [f"protonet_{mn}" for mn in ['ilsvrc_2012', 'aircraft', 'cu_birds', 'quickdraw',
                                               'vgg_flower', 'omniglot', 'dtd']]

    window_len = 2000
    stride = 500
    rolling_loader = iter(roll_data(input_data=all_data, stride=stride, window_size=window_len))
    acc_all, conf_all, base_all = [], [], []
    for i, in_data in enumerate(rolling_loader):
        exp_result = train_fusion(in_data=in_data[:, :-1], in_labels=in_data[:, -1], n_epochs=150, verbose=False)
        acc_all.append(exp_result["test_acc"])
        conf_all.append(exp_result["test_conf"])
        base_all.append(get_base_pred(in_data[-stride:], model_names, n_way=5))

    acc_all, conf_all, base_all = np.array(acc_all), np.array(conf_all), np.array(base_all)
    np.save("acc_all_rolling.npy", acc_all)
    np.save("conf_all_rolling.npy", conf_all)
    np.save("base_all.npy", base_all)

    plt.figure()
    plt.plot(acc_all, label="FusionShot")
    for i, mn in enumerate([model_names[0]]):
        plt.plot(base_all[:, i], label=mn)
    plt.ylim(0, 100)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()
