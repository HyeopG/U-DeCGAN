    ##
import os

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import *
from dataset import *
from util import *

def train(args):
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    mode = args.mode

    train_continue = args.train_continue

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float_)]

    ny = args.ny
    nx = args.nx
    in_channels = args.in_channels
    out_channels = args.out_channels
    nker = args.nker

    wgt = args.wgt
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("train_continue: %s" % train_continue)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("data_dir: %s" % data_dir)
    print("ckpt_dir: %s" % ckpt_dir)
    print("log_dir: %s" % log_dir)
    print("result_dir: %s" % result_dir)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("ny: %s" % ny)
    print("nx: %s" % nx)
    print("in_channels: %s" % in_channels)
    print("out_channels: %s" % out_channels)
    print("kernel: %s" % nker)

    print("wgt: %s" % wgt)
    print("norm: %s" % norm)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("device: %s" % device)

    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir_train, "png"))


    ## 네트워크 학습하기
    transform_train = transforms.Compose([Resize(shape=(ny, nx, out_channels)), Normalization(mean=0.5, std=0.5)])

    dataset_train = Dataset(data_dir=data_dir, transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    num_data_train = len(dataset_train)
    num_batch_train = np.ceil(num_data_train / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=out_channels, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=out_channels, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    if network == "U-DeCGAN":
        netG = UNet(in_channels=in_channels, out_channels=out_channels, nker=nker, norm=norm, learning_type=learning_type).to(device)
        netD = Discriminator(in_channels=out_channels, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)
    ## 손실함수 정의하기
    #fn_loss = nn.BCEWithLogitsLoss().to(device)

    fn_mse = nn.MSELoss().to(device)
    fn_gan = nn.BCELoss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    #fn_class = lambda  x: 1.0 * (x>0.5)

    cmap = None

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    ## 네트워크 학습하기
    st_epoch = 0

    if train_continue:
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        netG.train()
        netD.train()

        loss_G_mse_arr = []
        loss_G_gan_arr = []
        loss_D_real_arr = []
        loss_D_fake_arr = []

        for batch, data in enumerate(loader_train, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)
            #input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

            output = netG(input)

            # backward netD
            set_requires_grad(netD, True)
            optimD.zero_grad()

            pred_real = netD(label)
            pred_fake = netD(output.detach())

            loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
            loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optimD.step()

            # backward netG
            set_requires_grad(netD, False)
            optimG.zero_grad()

            pred_fake = netD(output)

            loss_G_mse = fn_mse(output, label)
            loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
            loss_G = loss_G_gan + loss_G_mse * wgt
            loss_G.backward()
            optimG.step()

            #
            loss_G_mse_arr += [loss_G_mse.item()]
            loss_G_gan_arr += [loss_G_gan.item()]
            loss_D_real_arr += [loss_D_real.item()]
            loss_D_fake_arr += [loss_D_fake.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                    "GEN MSE %.4f | GEN GAN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
                  (epoch, num_epoch, batch, num_batch_train,
                   np.mean(loss_G_mse_arr), np.mean(loss_G_gan_arr), np.mean(loss_D_real_arr), np.mean(loss_D_fake_arr)))

            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

            input = np.clip(input, a_min=0, a_max=1)
            label = np.clip(label, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            id = num_batch_train * (epoch - 1) + batch

            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0].squeeze(), cmap="gray")
            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap="gray")
            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap="gray")

            # writer_train.add_image('output', output, id, dataformats='NHWC')

        writer_train.add_scalar('loss_G_mse', np.mean(loss_G_mse_arr), epoch)
        writer_train.add_scalar('loss_G_gan', np.mean(loss_G_gan_arr), epoch)
        writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_arr), epoch)
        writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_arr), epoch)

        if epoch % 10 == 0:
            save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

    writer_train.close()


def test(args):
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    mode = args.mode

    train_continue = args.train_continue

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float_)]

    ny = args.ny
    nx = args.nx
    in_channels = args.in_channels
    out_channels = args.out_channels
    nker = args.nker

    wgt = args.wgt
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("train_continue: %s" % train_continue)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("data_dir: %s" % data_dir)
    print("ckpt_dir: %s" % ckpt_dir)
    print("log_dir: %s" % log_dir)
    print("result_dir: %s" % result_dir)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("ny: %s" % ny)
    print("nx: %s" % nx)
    print("in_channels: %s" % in_channels)
    print("out_channels: %s" % out_channels)
    print("kernel: %s" % nker)

    print("wgt: %s" % wgt)
    print("norm: %s" % norm)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("device: %s" % device)

    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, "png"))

    ## 네트워크 학습하기
    transform_test = transforms.Compose([Resize(shape=(ny, nx, out_channels)), Normalization(mean=0.5, std=0.5)])

    dataset_test = Dataset(data_dir=data_dir, transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=out_channels, nker=nker).to(device)
        netD = Discriminator(in_channels=in_channels, out_channels=1, nker=nker).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    if network == "U-DeCGAN":
        netG = UNet(in_channels=in_channels, out_channels=out_channels, nker=nker, norm=norm, learning_type=learning_type).to(device)
        netD = Discriminator(in_channels=out_channels, out_channels=1, nker=nker).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    fn_mse = nn.MSELoss().to(device)
    fn_gan = nn.BCELoss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    # fn_class = lambda  x: 1.0 * (x>0.5)

    cmap = None

    ## 네트워크 학습하기
    netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG,
                                                optimD=optimD)

    with torch.no_grad():
        netG.eval()

        loss_G_mse_test = []

        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            #input = data['input'].to(device)
            #input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

            output = netG(label)
            # output = netG(input)
            # loss_G_mse = fn_mse(output, label)
            #
            # loss_G_mse_test += [loss_G_mse.item()]
            #
            # print("TEST: BATCH %.4f / %.4f | GEN MSE %.4f" %
            #       (batch, num_batch_test, np.mean(loss_G_mse_test)))

            #input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

            for j in range(label.shape[0]):
                id = batch_size * (batch-1) + j

                #input_ = input[j]
                label_ = label[j]
                output_ = output[j]

                #input_ = np.clip(input_, a_min=0, a_max=1)
                label_ = np.clip(label_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                #plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_.squeeze(), cmap="gray")
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_.squeeze(), cmap="gray")
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_.squeeze(), cmap="gray")

        #print("AVERAGE TEST: GEN L1 %.4f" % np.mean(loss_G_mse_test))
