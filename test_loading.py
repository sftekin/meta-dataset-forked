from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
from collections import Counter
import gin
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
import methods.backbone as backbone
from methods.protonet import ProtoNet
from methods.ResNet import resnet18

model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101)

BASE_PATH = '/media/selim/data/metadataset/records'
GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/data_config.gin'
# 2
gin.parse_config_file(GIN_FILE_PATH)

NUM_WAYS = 5
NUM_SUPPORT = 5
NUM_QUERY = 15
# ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
#                 'omniglot', 'quickdraw', 'vgg_flower']
ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'omniglot', 'quickdraw', 'ilsvrc_2012']
N_EPISODES = 100
N_EPOCH = 700
CHECKPOINT_DIR = "checkpoints"
SAVE_FREQ = 10
VARIABLE_WAYS_SHOT = False


def iterate_dataset(dataset, n):
    if not tf.executing_eagerly():
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            for idx in range(n):
                yield idx, sess.run(next_element)
    else:
        for idx, episode in enumerate(dataset):
            if idx == n:
                break
            yield idx, episode


def train_loop(model, epoch, train_loader, optimizer, dataset_spec):
    print_freq = 50
    avg_loss = 0
    for i, (episode, _) in iterate_dataset(train_loader, N_EPISODES):
        optimizer.zero_grad()
        episode = [torch.from_numpy(a.numpy()) for a in episode]
        if VARIABLE_WAYS_SHOT:
            loss = model.set_forward_loss(episode, y=episode[4], variable=True)
        else:
            im_sup, im_query = episode[0].permute(0, 3, 1, 2), episode[3].permute(0, 3, 1, 2)
            im_sup = im_sup.reshape(NUM_WAYS, NUM_SUPPORT, *im_sup.shape[1:])
            im_query = im_query.reshape(NUM_WAYS, NUM_QUERY, *im_query.shape[1:])
            x = torch.cat([im_sup, im_query], dim=1)
            loss = model.set_forward_loss(x, variable=False)
        loss.backward()
        optimizer.step()
        avg_loss = avg_loss + loss.item()

        if i % print_freq == 0:
            print(f'From source {dataset_spec.name}, '
                  f'Epoch {epoch} | Batch {i}/{N_EPISODES} | Loss {avg_loss / float(i + 1)}')

    # for idx, (episode, source_id) in iterate_dataset(dataset_train, N_EPISODES):
    #     print('Episode id: %d from source %s' % (idx, train_loader.dataset_spec.name))
    #     episode = [torch.from_numpy(a.numpy()) for a in episode]
    #     y = episode[4]
    #     x = torch.cat([episode[0].permute(0, 3, 1, 2), episode[3].permute(0, 3, 1, 2)])
    #
    #     a = model.set_forward(x)

    # z_sup = model.feature.forward(im_sup)
    # z_que = model.feature.forward(im_query)
    # print()


def test_loop(model, test_loader, record=None):
    correct = 0
    count = 0
    acc_all = []

    for i, (episode, _) in iterate_dataset(test_loader, N_EPISODES):
        episode = [torch.from_numpy(a.numpy()) for a in episode]
        if VARIABLE_WAYS_SHOT:
            correct_this, count_this = model.correct(episode, y=episode[4], variable=True)
        else:
            im_sup, im_query = episode[0].permute(0, 3, 1, 2), episode[3].permute(0, 3, 1, 2)
            im_sup = im_sup.reshape(NUM_WAYS, NUM_SUPPORT, *im_sup.shape[1:])
            im_query = im_query.reshape(NUM_WAYS, NUM_QUERY, *im_query.shape[1:])
            x = torch.cat([im_sup, im_query], dim=1)
            correct_this, count_this = model.correct(x, variable=False)
        acc_all.append(correct_this / count_this * 100)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (N_EPISODES, acc_mean, 1.96 * acc_std / np.sqrt(N_EPISODES)))

    return acc_mean


def run(dataset_name):
    model_name = "ResNet18"
    save_file_name = f"protonet_{model_name}_variable" if VARIABLE_WAYS_SHOT \
        else f"protonet_{model_name}_{NUM_SUPPORT}shot_{NUM_WAYS}way"
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, dataset_name, save_file_name)
    if not os.path.isdir(checkpoint_dir):
        exp_num = 0
    else:
        exp_num = len(glob.glob(os.path.join(checkpoint_dir, "exp_*")))
    checkpoint_dir = os.path.join(checkpoint_dir, f"exp_{exp_num}")
    os.makedirs(checkpoint_dir)

    params = dict(dataset_name=dataset_name, model_name=model_name, variable_way_shot=VARIABLE_WAYS_SHOT,
                  checkpoint_dir=checkpoint_dir, n_ways=NUM_WAYS, n_shot=NUM_SUPPORT, n_query=NUM_QUERY,
                  save_freq=SAVE_FREQ)

    if dataset_name not in ALL_DATASETS:
        raise KeyError("not extracted")

    dataset_records_path = os.path.join(BASE_PATH, dataset_name)
    dataset_spec = [dataset_spec_lib.load_dataset_spec(dataset_records_path)]

    if VARIABLE_WAYS_SHOT:
        episode_descr_config = config.EpisodeDescriptionConfig(
            num_query=None, num_support=None, num_ways=None, max_support_set_size=100,
            min_ways=5, max_num_query=5)
    else:
        episode_descr_config = config.EpisodeDescriptionConfig(
            num_ways=NUM_WAYS, num_support=NUM_SUPPORT, num_query=NUM_QUERY)

    if dataset_name == "omniglot" and VARIABLE_WAYS_SHOT:
        use_bilevel_ontology_list = [True]
    else:
        use_bilevel_ontology_list = [False]

    if dataset_name == "ilsvrc_2012" and VARIABLE_WAYS_SHOT:
        use_dag_ontology_list = [True]
    else:
        use_dag_ontology_list = [False]

    train_split = learning_spec.Split.TRAIN
    base_loader = pipeline.make_multisource_episode_pipeline(
        dataset_spec_list=dataset_spec, use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list, split=train_split,
        image_size=84, episode_descr_config=episode_descr_config)

    val_loader = pipeline.make_multisource_episode_pipeline(
        dataset_spec_list=dataset_spec, use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list, split=learning_spec.Split.VALID,
        image_size=84, episode_descr_config=episode_descr_config)

    feature_model = resnet18
    model = ProtoNet(feature_model, n_way=NUM_WAYS, n_support=NUM_SUPPORT, n_query=NUM_QUERY)
    model.cuda()

    print(params)
    optimizer = torch.optim.Adam(model.parameters())
    max_acc = -99
    tolerance = 50
    patience = 0
    for epoch in range(N_EPOCH):
        model.train()
        train_loop(model, epoch, base_loader, optimizer, dataset_spec[0])
        model.eval()

        acc = test_loop(model, val_loader)
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            patience = 0
        else:
            patience += 1

        if (epoch % SAVE_FREQ == 0) or (epoch == N_EPOCH - 1):
            outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if patience >=tolerance:
            outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            print(f"No improvement is seen on validation "
                  f"accuracy since {tolerance} epoch, early stopping...")
            break


if __name__ == '__main__':
    # ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
    #                 'omniglot', 'quickdraw', 'vgg_flower']
    run(dataset_name="ilsvrc_2012")
