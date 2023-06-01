# Illuminance Distribution Control using VAE Encoder and DCGAN

### Illuminance Distribution Control using VanillaVAEEncoder - DCGAN Image Generation - Python, PyTorch
* This project is being performed in collaboration with Colin Acton and SangYoon Back at 
MACS Lab advised by Dr. Xu Chen.
* Implemented Vanilla VAE Encoder to create latent embeddings based on depth map and
normal map.
* The latent space embedding combined with change in intensities of the LEDs 
surrounding the camera is given as input to Generator of DCGAN to generate change in
illuminance images.
* A combination of Log_MSELoss for VAE Encoder and BCELoss for DCGAN was used to update
the model parameters for each iteration.
* Trained the model using custom rendered dataset using Blender for 10 epochs and  below 
is the gif of transformation of generated images with each epoch of training.
* Currently improving the loss functions and model architecture to get more reliable
outputs.
![alt text](https://github.com/mkpvasu/Illuminance-Distribution-Control-using-VAE-and-GAN/blob/main/idc_with_vae_and_dcgan/idc_vae_dcgan_animation.gif)


### DCGAN - Python, PyTorch
* Implemented DCGAN - generator and discriminator architecture with training following 
pytorch tutorials.
* Written in Object-Oriented Design using classes rather than jupyter notebook used 
in the tutorial.
* Trained the model using celeba-dialog dataset having 30k 1024*1024 pixel images (
resized to 64*64 images) for 20 epochs and below is the gif of transformation of 
generated images with each epoch of training.
![alt text](https://github.com/mkpvasu/Illuminance-Distribution-Control-using-VAE-and-GAN/blob/main/dcgan_pytorch/dcgan_animation.gif)