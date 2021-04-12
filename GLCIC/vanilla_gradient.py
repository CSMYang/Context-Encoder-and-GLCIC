from torchvision import transforms
from PIL import Image
import torch
from matplotlib import pyplot as plt
from model_pretrained import ContextDiscriminator
import numpy as np


def visualize_saliency_map(img_path, masked_img, input_width, input_height, model):
    """
    This function shows the saliency map of an image
    :param img_path: a string of the path to the image for visualization
    :param input_width: input height to the model
    :param input_height: input width to the model
    :param model: the model used for image classification. In our case, it will be a discriminator
    """
    image = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(input_width),
        transforms.CenterCrop(input_width),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = transform(image)
    image.reshape(1, 3, input_height, input_width)
    image = torch.unsqueeze(image, dim=0)
    image = image.cuda()
    image.retain_grad()
    image.requires_grad_()

    masked_img = masked_img.cuda()
    masked_img.retain_grad()
    masked_img.requires_grad_()

    output = model((masked_img, image))

    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()

    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    saliency = saliency.reshape(input_height, input_width)

    saliency_2, _ = torch.max(masked_img.grad.data.abs(), dim=1)

    temp_img = image[0].reshape(-1, input_height, input_width)
    temp_image_1 = torch.ones((3, 64, 64), dtype=int)
    temp_image_1 = temp_img + 1
    temp_image_1 *= 255/2

    fig, ax = plt.subplots(2, 2)

    ax[0][0].imshow(temp_image_1.cpu().detach(
    ).numpy().astype(np.int).transpose(1, 2, 0))
    ax[0][0].axis('off')
    ax[0][1].imshow(saliency.cpu(), cmap='hot')
    ax[0][1].axis('off')

    ax[1][0].imshow(masked_img[0].cpu().detach(
    ).numpy().astype(np.float32).transpose(1, 2, 0))
    ax[1][0].axis('off')
    ax[1][1].imshow(saliency_2[0].cpu(), cmap='hot')
    ax[1][1].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass
    # CD = ContextDiscriminator(local_input_shape=(3, 96, 96),
    #                           global_input_shape=(
    #     3, 160, 160),
    #     arc='celeba')
    # CD.load_state_dict(
    #     state_dict=torch.load("GLCIC\pretrained_model_cd"))

    # visualize_saliency_map("GLCIC\\result.jpg", 160, 160, CD)
