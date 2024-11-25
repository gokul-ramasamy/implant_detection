import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt

from vit_pytorch.vit import Transformer
from tqdm import tqdm

class MAE_Recon(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        patch_size = 8
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.patch_size = patch_size



    def reconstruct_image(self, patches, model_input, masked_indices=None, pred_pixel_values=None, patch_size=8):
        """
        Reconstructs the image given patches. Can also reconstruct the masked image as well as the predicted image.
        To reconstruct the raw image from the patches, set masked_indices=None and pred_pixel_values=None. To reconstruct
        the masked image, set masked_indices= the masked_indices tensor created in the `forward` call. To reconstruct the
        predicted image, set masked_indices and pred_pixel_values = to their respective tensors created in the `forward` call.

        ARGS:
            patches (torch.Tensor): The raw patches (pre-patch embedding) generated for the given model input. Shape is
                (batch_size x num_patches x patch_size^2 * channels)
            model_input (torch.Tensor): The input images to the given model (batch_size x channels x height x width)
            mean (list[float]): An array representing the per-channel mean of the dataset used to
                denormalize the input and predicted pixels. (1 x channels)
            std (list[float]): An array representing the per-channel std of the dataset used to
                denormalize the input and predicted pixels. (1 x channels)
            masked_indices (torch.Tensor): The patch indices that are masked (batch_size x masking_ratio * num_patches)
            pred_pixel_values (torch.Tensor): The predicted pixel values for the patches that are masked (batch_size x masking_ratio * num_patches x patch_size^2 * channels)

        RETURN:
            reconstructed_image (torch.Tensor): Tensor containing the reconstructed image (batch_size x channels x height x width)
        """
        patches = patches.cpu()

        masked_indices_in = masked_indices is not None
        predicted_pixels_in = pred_pixel_values is not None

        if masked_indices_in:
            masked_indices = masked_indices.cpu()

        if predicted_pixels_in:
            pred_pixel_values = pred_pixel_values.cpu()

        patch_width = patch_height = self.patch_size
        reconstructed_image = patches.clone()

        if masked_indices_in or predicted_pixels_in:
            for i in range(reconstructed_image.shape[0]):
                if masked_indices_in and predicted_pixels_in:
                    reconstructed_image[i, masked_indices[i].cpu()] = pred_pixel_values[i, :].cpu().float()
                elif masked_indices_in:
                    reconstructed_image[i, masked_indices[i].cpu()] = 0

        invert_patch = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w=int(model_input.shape[3] / patch_width),
                                    h=int(model_input.shape[2] / patch_height), c=model_input.shape[1],
                                    p1=patch_height, p2=patch_width)

        reconstructed_image = invert_patch(reconstructed_image)

        # We are using a non-normalized image (so this is not required)
        # reconstructed_image = reconstructed_image.numpy().transpose(0, 2, 3, 1)
        # reconstructed_image *= np.array(std)
        # reconstructed_image += np.array(mean)

        return reconstructed_image

    # def forward(self, img):
    #     device = img.device

    #     # get patches
    #     patches = self.to_patch(img)
    #     batch, num_patches, *_ = patches.shape  ## Patches shape [batch_size, num_patches, 512, 512]

    #     # patch to encoder tokens and add positions

    #     tokens = self.patch_to_emb(patches)
    #     if self.encoder.pool == "cls":
    #         tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
    #     elif self.encoder.pool == "mean":
    #         tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

    #     # Getting all the indices (basically all the indices are unmasked)
    #     unmasked_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)

    #     # get the unmasked tokens to be encoded (all the tokens)
    #     batch_range = torch.arange(batch, device = device)[:, None]
    #     tokens = tokens[batch_range, unmasked_indices]

    #     ## get the patches to be masked for the final reconstruction loss (no patches are masked)
    #     masked_patches = patches[batch_range, unmasked_indices]

    #     ## attend with vision transformer
    #     encoded_tokens = self.encoder.transformer(tokens)

    #     ## project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

    #     decoder_tokens = self.enc_to_dec(encoded_tokens)

    #     ## reapply decoder position embedding to unmasked tokens

    #     unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

    #     ## repeat mask tokens for number of masked, and add the positions using the masked indices derived above

    #     # mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
    #     # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

    #     ## concat the masked tokens to the decoder tokens and attend with decoder
        
    #     decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
    #     decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
    #     # decoder_tokens[batch_range, masked_indices] = mask_tokens
    #     decoded_tokens = self.decoder(decoder_tokens)

    #     ## splice out the mask tokens and project to pixel values
    #     mask_tokens = decoded_tokens[batch_range, unmasked_indices]
    #     pred_pixel_values = self.to_pixels(mask_tokens)  ### Output shape

    #     # Getting the original image, masked image and reconstruction
    #     # original image
    #     original_image = self.reconstruct_image(patches=patches, model_input=img, patch_size=16)
    #     masked_image = original_image.clone()
    #     reconstructed_image = self.reconstruct_image(patches=patches, model_input=img, masked_indices=unmasked_indices, pred_pixel_values=pred_pixel_values, patch_size=16)
    #     # Calculating the reconstruction loss
    #     recon_loss = F.mse_loss(original_image, reconstructed_image.clip(0))
    #     reconstructed_image = reconstructed_image.clip(0)


    #     return recon_loss, original_image, masked_image, reconstructed_image


    def forward(self, img):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape  ## Patches shape [batch_size, num_patches, 512, 512]
        batch_range = torch.arange(batch, device = device)[:, None]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)

        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # print(masked_indices.shape)
        # print(unmasked_indices.shape)

        # print("OLDER")

        all_masked_indices = list()
        all_predicted_pixels = list()

        # Everything I guess rests here
        for i in range(0, rand_indices.shape[1], num_masked):
            masked_indices = rand_indices[:, i:i+num_masked]
            if i!= 0:
                unmasked_indices = torch.cat((rand_indices[:, :i], rand_indices[:, i+num_masked:]), axis=1)
            else:
                unmasked_indices = rand_indices[:, i+num_masked:]

            # get the unmasked tokens to be encoded
            unmasked_tokens = tokens[batch_range, unmasked_indices]

            # get the patches to be masked for the final reconstruction loss
            masked_patches = patches[batch_range, masked_indices]

            # attend with vision transformer

            encoded_tokens = self.encoder.transformer(unmasked_tokens)

            # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

            decoder_tokens = self.enc_to_dec(encoded_tokens)

            # reapply decoder position embedding to unmasked tokens

            unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

            # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

            mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = masked_indices.shape[1])  # Gokul made a change here
            mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

            # concat the masked tokens to the decoder tokens and attend with decoder
            
            decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
            decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
            decoder_tokens[batch_range, masked_indices] = mask_tokens
            decoded_tokens = self.decoder(decoder_tokens)

            # splice out the mask tokens and project to pixel values

            mask_tokens = decoded_tokens[batch_range, masked_indices]
            pred_pixel_values = self.to_pixels(mask_tokens)  ### Output shape

            # calculate reconstruction loss

            recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

            all_masked_indices.append(masked_indices)
            all_predicted_pixels.append(pred_pixel_values)

            # with torch.no_grad():
            #     torch.cuda.empty_cache()
            
            # unmasked_tokens = None
            # encoded_tokens = None
            # decoder_tokens = None
            # decoded_tokens = None
            



        final_indices = torch.cat(all_masked_indices, axis=1)
        final_predicted_pixels = torch.cat(all_predicted_pixels, axis=1)

        
        reconstructed_image = patches.clone().to(device)
        for i in range(reconstructed_image.shape[0]):
            reconstructed_image[i, final_indices[i].to(device)] = final_predicted_pixels[i, :].to(device).float()
        
        patch_width = patch_height = self.patch_size 
        model_input = img
        invert_patch = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w=int(model_input.shape[3] / patch_width),
                            h=int(model_input.shape[2] / patch_height), c=model_input.shape[1],
                            p1=patch_height, p2=patch_width)

        reconstructed_image = invert_patch(reconstructed_image)

        return img.detach().cpu().numpy(), reconstructed_image.detach().cpu().numpy()

        exit()


        # Getting the original image, masked image and reconstruction
        # original image
        original_image = self.reconstruct_image(patches=patches, model_input=img, patch_size=16)
        masked_image = self.reconstruct_image(patches=patches, model_input=img, masked_indices=masked_indices, patch_size=16)
        reconstructed_image = self.reconstruct_image(patches=patches, model_input=img, masked_indices=masked_indices, pred_pixel_values=pred_pixel_values, patch_size=16)

        # reconstructed_image = reconstructed_image.clip(0)
        # print(original_image.shape)
        # print(masked_image.shape)
        # print(reconstructed_image.shape)

        # fig, ax = plt.subplots(1, 3, figsize=(10,20))
        # fig.tight_layout()
        # ax[0].imshow(original_image.detach().cpu().numpy().squeeze(), cmap='gray')
        # ax[0].axis('off')
        # ax[1].imshow(masked_image.detach().cpu().numpy().squeeze(), cmap='gray')
        # ax[1].axis('off')
        # ax[2].imshow(reconstructed_image.detach().cpu().numpy().squeeze(), cmap='gray')
        # ax[2].axis('off')

        # plt.show()


        return recon_loss, original_image, masked_image, reconstructed_image
    
    




