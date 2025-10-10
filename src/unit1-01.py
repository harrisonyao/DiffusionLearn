import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import DDPMPipeline

""" 增加本地代理 """
import os
os.environ["http_proxy"] = "http://127.0.0.1:12334"
os.environ["https_proxy"] = "http://127.0.0.1:12334"


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


# Mac users may need device = 'mps' (untested)
# 正确的设备选择优先级：MPS > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


''' step1 test generate with prompt '''
'''
# Check out https://huggingface.co/sd-dreambooth-library for loads of models from the community
model_id = "sd-dreambooth-library/mr-potato-head"

# Load the pipeline
pipe = (StableDiffusionPipeline.from_pretrained(model_id,
                                                allow_pickle=False,
                                                use_safetensors=False,
                                                torch_dtype=torch.float16)
        .to(device))

prompt = "an abstract oil painting of sks mr potato head by picasso, white background"
image = pipe(prompt, num_inference_steps=100, guidance_scale=5.0).images[0]
image.show()

'''

# Load the butterfly pipeline
butterfly_pipeline = DDPMPipeline.from_pretrained(
    "johnowhitaker/ddpm-butterflies-32px"
).to(device)

# Create 8 images
images = butterfly_pipeline(batch_size=8).images

# View the result
make_grid(images)

''' step 2 use butter fly dataset to  '''
'''
from diffusers import DDPMPipeline

# Load the butterfly pipeline
butterfly_pipeline = DDPMPipeline.from_pretrained(
    "johnowhitaker/ddpm-butterflies-32px"
).to(device)

# Create 8 images
images = butterfly_pipeline(batch_size=8).images

# View the result
outimg = make_grid(images)
outimg.show()
'''

''' step 3 load dataset '''
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

# Or load images from a local folder
# dataset = load_dataset("imagefolder", data_dir="path/to/folder")

# We'll train on 32-pixel square images, but you can try larger sizes too
image_size = 32
# You can lower your batch size if you're running out of GPU memory
batch_size = 64

# Define data augmentations
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

xb = next(iter(train_dataloader))["images"].to(device)[:8]
print("X shape:", xb.shape)
show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)

# 初始化噪点生成器
from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
plt.legend(fontsize="x-large")

# 给图片增加噪点
timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(xb)
noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
print("Noisy X shape", noisy_xb.shape)
show_images(noisy_xb).resize((8 * 64, 64), resample=Image.NEAREST)

''' step 4 create unet model '''
from diffusers import UNet2DModel

# Create a model
model = UNet2DModel(
    sample_size=image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
model.to(device)

with torch.no_grad():
    model_prediction = model(noisy_xb, timesteps).sample
model_prediction.shape

''' step 5 create a trainning loop'''
# Set the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)
# # for step 6 running directly, commented this block
# # Training loop
# optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
#
# losses = []
#
# for epoch in range(30):
#     for step, batch in enumerate(train_dataloader):
#         clean_images = batch["images"].to(device)
#         # Sample noise to add to the images
#         noise = torch.randn(clean_images.shape).to(clean_images.device)
#         bs = clean_images.shape[0]
#
#         # Sample a random timestep for each image
#         timesteps = torch.randint(
#             0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
#         ).long()
#
#         # Add noise to the clean images according to the noise magnitude at each timestep
#         noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
#
#         # Get the model prediction
#         noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
#
#         # Calculate the loss
#         loss = F.mse_loss(noise_pred, noise)
#         loss.backward(loss)
#         losses.append(loss.item())
#
#         # Update the model parameters with the optimizer
#         optimizer.step()
#         optimizer.zero_grad()
#
#     if (epoch + 1) % 5 == 0:
#         loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
#         print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
#
# #prepare for the plot tool to print the loss
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# axs[0].plot(losses)
# axs[1].plot(np.log(losses))
# plt.show()


''' step 6 Generate images '''
# Uncomment to instead load the model I trained earlier:
model = butterfly_pipeline.unet

from diffusers import DDPMPipeline

image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
pipeline_output = image_pipe()
pipeline_output.images[0]

image_pipe.save_pretrained("harrison_butterfly_pipline")