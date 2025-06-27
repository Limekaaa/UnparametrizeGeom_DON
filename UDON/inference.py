
import torch
import UDON
import logging

import numpy as np

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default
    
def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        #latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
        #latent_code = torch.nn.Embedding(1, latent_size).cuda()
        latent_code = torch.nn.Embedding(1, latent_size).to(device)
        torch.nn.init.normal_(latent_code.weight, mean=0.0, std=stat)
    else:
        #latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
        latent = torch.normal(stat[0].detach(), stat[1].detach()).to(device)


    #latent.requires_grad = True # commented out by me

    #optimizer = torch.optim.Adam([latent], lr=lr) # commented out by me
    optimizer = torch.optim.Adam(latent_code.parameters(), lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()

        # Added by me _____________________________________________________________________
        if type(stat) == type(0.1):
            #latent = latent_code(torch.tensor([0]).cuda())
            latent = latent_code(torch.tensor([0]).to(device))

        # And of added by me _____________________________________________________________________
        """
        sdf_data = UDON.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        """
        sdf_data = UDON.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).to(device)


        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        #inputs = torch.cat([latent_inputs, xyz], 1).cuda()
        inputs = torch.cat([latent_inputs, xyz], 1).to(device)

        pred_sdf = decoder(inputs.float())
        """
        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs.float())
        """
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            print(f"loss: {loss.cpu().data.numpy()}")
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    print(f"Final loss: {loss_num}")
    #return loss_num, latent
    #return loss_num, latent_code(torch.tensor([0]).cuda())  # Return the latent code from the embedding
    return loss_num, latent_code(torch.tensor([0]).to(device))  # Return the latent code from the embedding