import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
from utils.data import LabeledData

import matplotlib.pyplot as plt
def getsim(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)
    model.eval()
    sums = 0
    num = 0
    counts = torch.zeros(3100)
    for i, (img, label) in enumerate(loader):
        model.eval()
        model.zero_grad()
        # print(label)
        img = img.to(device)

        out, latent_loss,sim = model(img)
        label_m = torch.mm(label,label.T).type(torch.bool)
        val,indc = (-sim[0]).topk(10)
        print(val,indc)
        diag_E = torch.eye(label_m.shape[0],dtype=bool)
        print(label_m[0][indc])
        # label_m[diag_E]=0
        # print(~label_m)
        num+=1
        # result = sim[label_m].type(torch.LongTensor)
        # for i in result:
        #     counts[i]+=1
        # print(counts)
    print(sums/num)
    # fig = plt.figure()
    # plt.bar(range(3100),counts,0.4,color="green")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("bar chart")
    
    
    
    # plt.savefig("barChart.jpg")

    # print(sums/num)
def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    model.eval()
    for i, (img, label) in enumerate(loader):
        model.eval()
        model.zero_grad()
        # print(label)
        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()
    if epoch+1 == 2:
        torch.save(model.state_dict(), f"/home/yangxv/vq-vae-2-pytorch/vqvae_560.pt")
def val(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        model.eval()
        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f"./test.png",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()
    return

def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(args.size),
    #         transforms.CenterCrop(args.size),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )

    # dataset = datasets.ImageFolder(args.path, transform=transform)
    # sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    # loader = DataLoader(
    #     dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    # )
    data = LabeledData('cifar10')
    loader, val_loader, _, database_loader = data.get_loaders(
    128, 4,shuffle_train=True, get_test=False)
    model = VQVAE().to(device)
    model.load_state_dict(torch.load("/home/yangxv/vq-vae-2-pytorch/vqvae_560.pt"))
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )
    # val(0, loader, model, optimizer, scheduler, device)
    for i in range(args.epoch):
        getsim(i, val_loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
