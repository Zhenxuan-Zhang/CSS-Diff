# This code is released under the CC BY-SA 4.0 license.

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visual_validation import validation
import wandb
import torch
import copy
#import monai
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
#Set W&B Sweep configuration
sweep_configuration = {
    "method": "grid",
    "name": "PSS-Diff",
    "parameters": {
        "fth": {"values": [0.25]},
        "depth": {"values": [12]},
        "n_layers_D": {"values":[3]},
        "structured_shape_iter": {"values": [0]}
}}

# Initialize sweep by passing in config
sweep_id = wandb.sweep(sweep=sweep_configuration, project = "sweep-PSS-Diff")

# if __name__ == '__main__':
def main():
    opt = TrainOptions().parse()   # get training options
    if not opt.wdb_disabled: 
        wandb.init(project="PSS-Diff", name=opt.name)
    
    # Overwrite parameters with wandb
    opt.fth = wandb.config.fth
    opt.depth = wandb.config.depth
    opt.n_layers_D = wandb.config.n_layers_D
    opt.structured_shape_iter = wandb.config.structured_shape_iter

    # Name for the experiment to save the checkpoints
    exp_name = f"{opt.model}_{opt.netG}_fth{opt.fth}_depth{opt.depth}_nld{opt.n_layers_D}_ssi{opt.structured_shape_iter}"
    exp_name = exp_name.replace(".","")
    opt.name = exp_name
    # import pprint
    # pprint.pprint(vars(opt))


    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    data_print = next(iter(dataset))

    print("A_path:", data_print['A_paths']) #data check
    print("B_path:", data_print['B_paths'])


    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'val'
    val_opt.serial_batches = True
    val_dataset = create_dataset(val_opt)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations 
    best_fid = 80
    best_psnr = 10
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.train()
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            # 🔁 NEW: Validate and save best model every opt.val_freq steps
            if total_iters % 3000 == 0:
                print(f"[Validation] Running at iter {total_iters}...")

                if 'unaligned' in opt.dataset_mode:
                    fid, inception_score, ms_ssim = validation(val_dataset, model, val_opt, device=model.device)
                    wandb.log({'fid': fid, 'inception_score': inception_score, 'ms_ssim': ms_ssim}, step=total_iters)

                    if fid < best_fid:
                        print(f"✔ Best Model (FID = {fid:.4f}) at iter {total_iters}")
                        model.save_networks('best')
                        best_fid = fid
                else:
                    ssim, psnr = validation(val_dataset, model, val_opt, device=model.device)
                    wandb.log({'psnr': psnr, 'ssim': ssim}, step=total_iters)

                    if psnr > best_psnr:
                        print(f"✔ Best Model (PSNR = {psnr:.2f}) at iter {total_iters}")
                        model.save_networks('best')
                        best_psnr = psnr
            iter_data_time = time.time() 
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)       
        # for i, data in enumerate(dataset):# inner loop within one epoch
        #     # print("A_path:", data['A_paths'])
        #     # print("B_path:", data['B_paths'])
        #     # print("Shape of A:", data['A'].shape)
        #     # print("Shape of B:", data['B'].shape)
        #     # print("A min/max:", data['A'].min().item(), "/", data['A'].max().item())
        #     # print("B min/max:", data['B'].min().item(), "/", data['B'].max().item())
        #     # break  # exit the loop after first batch
        #     iter_start_time = time.time()  # timer for computation per iteration                
        #     if total_iters % opt.print_freq == 0:
        #         t_data = iter_start_time - iter_data_time
        #     total_iters += opt.batch_size
        #     epoch_iter += opt.batch_size
        #     model.train()
        #     model.set_input(data)         # unpack data from dataset and apply preprocessing
        #     model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        #     if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
        #         save_result = total_iters % opt.update_html_freq == 0
        #         model.compute_visuals()
        #         visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        #     if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
        #         losses = model.get_current_losses()
        #         t_comp = (time.time() - iter_start_time) / opt.batch_size
        #         visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
        #         if opt.display_id > 0:
        #             visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        #     if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
        #         print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        #         save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        #         model.save_networks(save_suffix)
                

        #     iter_data_time = time.time()         
        # if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)
        #     #Validate the model
        #     # Perform validation
        #     if 'unaligned' in opt.dataset_mode:
        #         fid, inception_score, ms_ssim = validation(val_dataset, model, val_opt, device=model.device)
        #         if fid < best_fid:
        #             print(f"Best Model with FID = {fid:.4f}")
        #             model.save_networks('best')
        #             best_fid = fid
        #     else:
        #         ssim, psnr = validation(val_dataset, model, val_opt, device=model.device)
        #         if psnr > best_psnr:
        #             print(f"Best Model with PSNR = {psnr:.2f}")
        #             model.save_networks('best')
        #             best_psnr = psnr

        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


# Initialize sweep by passing in config.
wandb.agent(sweep_id,function = main)

