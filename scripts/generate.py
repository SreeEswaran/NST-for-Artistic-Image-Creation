import torch
from torchvision import transforms, models
from PIL import Image
import argparse
import os

def load_image(img_path, transform=None):
    image = Image.open(img_path).convert('RGB')
    if transform:
        image = transform(image).unsqueeze(0)
    return image

def main(content_path, style_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = load_image(content_path, transform).to(device)
    style_image = load_image(style_path, transform).to(device)

    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_weight = 1e5
    style_weight = 1e10

    input_img = content_image.clone()

    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

    def closure():
        input_img.data.clamp_(0, 255)

        optimizer.zero_grad()

        content_loss = 0
        style_loss = 0

        for name, layer in vgg._modules.items():
            input_img = layer(input_img)
            if name in content_layers:
                content_loss += torch.mean((input_img - content_image) ** 2)
            if name in style_layers:
                style_loss += torch.mean((input_img - style_image) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()

        return total_loss

    optimizer.step(closure)

    input_img.data.clamp_(0, 255)
    stylized_image = input_img.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()

    os.makedirs(output_path, exist_ok=True)
    stylized_image = Image.fromarray(stylized_image.astype('uint8'))
    stylized_image.save(os.path.join(output_path, 'stylized_image.jpg'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate stylized images using Neural Style Transfer model.')
    parser.add_argument('--content_path', type=str, required=True, help='Path to content image')
    parser.add_argument('--style_path', type=str, required=True, help='Path to style image')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save stylized image')
    args = parser.parse_args()
    main(args.content_path, args.style_path, args.output_path)
