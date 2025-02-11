import torch
import time
import copy
import torch.nn.functional as F
# from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import rgb_to_grayscale
from piqa import SSIM
from matplotlib import pyplot as plt

train = True

def vit_model_train(model, ssim, num_epochs, dataloaders, device, 
                optimizer, scheduler, tf_writer,checkpoint_paths, es, reg_weight_mse, reg_weight_ssim):

    # To keep track of time
    since = time.time()

    # Best model weights and the best accuracy
    best_model_cvae_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print("\nEpoch Number - {}\n".format(epoch))
        epoch_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to train
                model.train()
            else:
                model.eval()

            iter_total_loss = 0.0 # Loss per iteration
            iter_mse_loss = 0.0 # Loss per iteration
            iter_ssim_loss = 0.0 # Loss per iteration

            for inputs in dataloaders[phase]:
                # Moving the inputs and the labels to device (CUDA or CPU)
                image = inputs[0].float().to(device) # check the size of the image here

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward Propagation
                with torch.set_grad_enabled(phase=='train'):
                    # Training
                    # Loss
                    mse_loss, original_image, masked_image, reconstructed_image = model(image)
                    # Getting the SSIM loss
                    X = rgb_to_grayscale(reconstructed_image.clone())
                    X2 = X.view(X.shape[0],X.shape[1],-1)
                    X2 -= X2.min(2, keepdim=True)[0]
                    X2 /= X2.max(2, keepdim=True)[0]
                    X2 = X2.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])

                    
                    ssim_loss = ssim(X2, rgb_to_grayscale(original_image))
                    total_loss = reg_weight_mse*mse_loss + reg_weight_ssim*ssim_loss

                    iter_mse_loss += mse_loss.item()*inputs[0].size(0)
                    iter_ssim_loss += ssim_loss.item()*inputs[0].size(0)
                    iter_total_loss += total_loss.item()*inputs[0].size(0)

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()
                

            epoch_mse_loss = iter_mse_loss/len(dataloaders[phase].dataset)
            epoch_ssim_loss = iter_ssim_loss/len(dataloaders[phase].dataset)
            epoch_total_loss = iter_total_loss/len(dataloaders[phase].dataset)
 
            print(f">> {phase} loss = {epoch_total_loss}")

            # Updating the scheduler
            if phase == 'val':
                scheduler.step(epoch_total_loss)
                lr = optimizer.param_groups[0]['lr']
                # Tf-writer: Learning Rates
                tf_writer.add_scalar('learning rate', lr, epoch)
            
            # Adding the input images
            tf_writer.add_images(phase+'_input', original_image, epoch)
            # Adding the masked images
            tf_writer.add_images(phase+'_masked_input', masked_image, epoch)
            # Reconstructed images
            tf_writer.add_images(phase+'_output', reconstructed_image, epoch)
            # Adding scalars
            tf_writer.add_scalar(phase+'_total_loss', epoch_total_loss, epoch)
            tf_writer.add_scalar(phase+'_mse_loss', epoch_mse_loss, epoch)
            tf_writer.add_scalar(phase+'_ssim_loss', epoch_ssim_loss, epoch)

            # Saving checkpoints after every ten epochs (pass in the frequency as a parameter)
            if epoch%10 == 0:
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_total_loss,  
                }, checkpoint_paths+'epoch_'+str(epoch)+'.pth')
        
        print("\n Time taken = {} s\n".format(round(time.time() - epoch_time, 2)))

        # Early Stopping
        # if phase == 'val':
        #     if es.step(epoch_total_loss):
        #         print("\n --- EARLY STOPPING --- \n")
        #         break
    
    time_elapsed = time.time() - since
    print("\nTraining complete in {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

    return None




