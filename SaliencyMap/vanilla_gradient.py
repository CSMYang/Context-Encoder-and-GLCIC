from torchvision import transforms
from PIL import Image
import torch
from matplotlib import pyplot as plt


def visualize_saliency_map(img_path, input_width, input_height, model, model_state_dict):
    """
    This function shows the saliency map of an image
    :param img_path: a string of the path to the image for visualization
    :param input_width: input height to the model
    :param input_height: input width to the model
    :param model: the model used for image classification. In our case, it will be a discriminator
    :param model_state_dict: the state dict for the model.
    """
    image = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(input_width),
        transforms.CenterCrop(input_width),
        transforms.Normalize(),
        transforms.ToTensor
    ])

    image = transform(image)
    image.reshape(1, 3, input_height, input_width)
    image = image.cuda()
    image.requires_grad_()

    output = model(image)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    saliency = saliency.reshape(input_height, input_width)

    image = image.reshape(-1, input_height, input_width)
    fig, ax = plt.subplot(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()