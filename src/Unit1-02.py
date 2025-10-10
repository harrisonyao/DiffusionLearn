# 导入必要的库
from diffusers import DiffusionPipeline  # 从diffusers库中导入DiffusionPipeline类
import os  # 导入os库，用于操作系统环境变量
import torch
from diffusers import DDPMPipeline, DDPMScheduler,UNet2DModel
from PIL import Image
import numpy as np

# 设置代理服务器，用于网络请求
os.environ["http_proxy"] = "http://127.0.0.1:12334"  # 设置HTTP代理
os.environ["https_proxy"] = "http://127.0.0.1:12334"  # 设置HTTPS代理
os.environ["CURL_CA_BUNDLE"] = ""


print("mps is available: ",torch.backends.mps.is_available())  # 应返回 True
print("mps is built: ",torch.backends.mps.is_built())      # 应返回 True

# # 从预训练模型加载扩散模型管道
# # generator = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")  # 加载预训练的文本到图像生成模型
# generator = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
#
# generator.to("mps") # for macbook m serial chipset
#
# image = generator("An image of squirrel in Picasso style").images[0]
#
# image.save("../output/image_of_squirrel_painting.png")
#
#


'''
1. load model and scheduler
'''
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("mps")

''' 
2. set the number of timesteps ro run the denoising process
'''
scheduler.set_timesteps(50)

'''
3. after set the timesteps, then run the step for create tensor with evenly spaced elements in it. Each element
corresponds to a timestep at which the model denoises an image.
'''
scheduler.timesteps

'''
4.create some random noise with the same shape as the desired output. this is creating a blank image with random noise likely
'''
sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("mps")

'''
5. write a loop to iterate over the timesteps. At each timestep, the model does a UNet2DModel.forward() pass and returns
the noisy residual(残余). The scheduler's step() method takes the noisy residual, timestep, and input and it predicts the image
ad the previous timestep. This output beconmes the next input to the model in the denoising loop, and it'll repeat
until it reaches the end of the timesteps aray.
'''
input = noise
for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample =  scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample

'''
6. the final output is the denoised image, so we need to convert it into an regular image format.
'''
image = (input / 2 + 0.5).clamp(0, 1)
print(f"After clamping, image shape: {image.shape}")
image = image.cpu().permute(0,2,3,1).numpy()[0]
image = (image * 255).astype(np.uint8)
print(f"After permute and numpy, image shape: {image.shape}")
image_pil = Image.fromarray(image)
print(f"After indexing, image shape: {image.shape}, dtype: {image.dtype}")
image_pil.show()