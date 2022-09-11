# the NST framework (including the NST functionalities documentation) is taken from
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html,
# which is implementing the algorithm developed by A. Gatys, Alexander S. Ecker and Matthias Bethge.
# Most of our (Amit, Ofek) added functionalities are contained in functions:
# get_objects_mask(), calc_distances_from_mask(), calculate_pixel_distances_from_objects(),
# run_style_transfer(), preprocessing_and_nst()
# The last 3 boxes are abstractions made for demo/submission purposes

# we recommend viewing the code and running it on colab notebook:
# https://colab.research.google.com/drive/1swoqs9qqGAkY_WpMC8PHYuzqg5TBXIQg#scrollTo=r_4icNHu4TZA

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import copy
import os
from scipy import ndimage
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# desired size of the output image
imsize = 700


def image_loader(style_image, style_image2, background_style_image, content_image, loader):
    style_image = Image.open(style_image)
    style_image2 = Image.open(style_image2)
    background_style_image = Image.open(background_style_image)
    content_image = Image.open(content_image)
    style_image = style_image.resize(content_image.size)
    style_image2 = style_image2.resize(content_image.size)
    background_style_image = background_style_image.resize(content_image.size)
    # fake batch dimension required to fit network's input dimensions
    style_image = loader(style_image).unsqueeze(0)
    style_image2 = loader(style_image2).unsqueeze(0)
    background_style_image = loader(background_style_image).unsqueeze(0)
    content_image = loader(content_image).unsqueeze(0)
    return style_image.to(device, torch.float), style_image2.to(device, torch.float), background_style_image.to(device,
                                                                                                                torch.float), content_image.to(
        device, torch.float)


def imshow(tensor, unloader, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, style_img2, background_style_img,
                               content_img,
                               content_layers=['conv_5'],
                               style_layers=['conv_1', 'conv_2', 'conv_3',
                                             'conv_4', 'conv_5']):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses1 = []
    style_losses2 = []
    background_style_losses = []
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style1_loss_{}".format(i), style_loss)
            style_losses1.append(style_loss)
            # 2nd style
            target_feature = model(style_img2).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style2_loss_{}".format(i), style_loss)
            style_losses2.append(style_loss)
            # background style
            target_feature = model(background_style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("background_style_loss_{}".format(i), style_loss)
            background_style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses1, style_losses2, background_style_losses, content_losses


def get_input_optimizer(input_img, lr):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img], lr=lr)
    return optimizer


def save_output(input_img, unloader, filename):
    image = input_img.cpu().clone()
    with torch.no_grad():
        input_img.clamp_(0, 1)
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(filename)


# calculate for each pixel its euclidean distance from each of the two main objects in the image
def calc_distances_from_mask(mask):
    flipped_mask = 1 - mask
    distances_from_mask_obj1 = torch.tensor(ndimage.distance_transform_edt(flipped_mask[:, :, 0].cpu()),
                                            requires_grad=False).to(device, torch.float)
    distances_from_mask_obj2 = torch.tensor(ndimage.distance_transform_edt(flipped_mask[:, :, 1].cpu()),
                                            requires_grad=False).to(device, torch.float)
    return torch.concat((distances_from_mask_obj1.unsqueeze(-1), distances_from_mask_obj2.unsqueeze(-1)), dim=-1)


# detect and mask the two most important objects in the image
def get_objects_mask(content_img, unloader, masks_directory):
    # download the segmentation model
    seg_model = models.detection.maskrcnn_resnet50_fpn(MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(
        device).eval().requires_grad_(False)
    content_seg = seg_model(content_img)[0]
    boxes, masks = content_seg['boxes'], content_seg['masks']
    boxes_squares = torch.tensor(
        [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in range(boxes.shape[0])])
    # pick two largest objects (can be extended to an arbitrary amount of largest objects available)
    max_squares, largest_objects_indices = boxes_squares.topk(2)
    first_obj_mask, second_obj_mask = masks[largest_objects_indices[0]], masks[largest_objects_indices[1]]
    # hard-mask the objects
    first_obj_mask[first_obj_mask > 0.5] = 1.0
    first_obj_mask[first_obj_mask <= 0.5] = 0.0
    second_obj_mask[second_obj_mask > 0.5] = 1.0
    second_obj_mask[second_obj_mask <= 0.5] = 0.0
    objects_masks_img = content_img * (first_obj_mask + second_obj_mask)
    plt.figure()
    imshow(objects_masks_img, unloader, title='Objects Segmentation Masks')
    save_output(objects_masks_img, unloader, os.path.join(masks_directory, "masked_objects.jpg"))
    stacked_mask = torch.concat((first_obj_mask.unsqueeze(-1), second_obj_mask.unsqueeze(-1)), dim=-1).squeeze(0)
    return stacked_mask


# derive a specific weight for each combination of pixel+style, based on the
# pixel's distance from the objects in the image
def calculate_pixel_distances_from_objects_and_convert_to_weights(input_img, unloader, masks_directory,
                                                                  object_styles_impact_on_bg=4):
    objects_mask = get_objects_mask(input_img, unloader, masks_directory)
    distances = calc_distances_from_mask(objects_mask)
    # convert distances to weights - that will be used as coefficients to the style's gradient on the pixel
    required_style_impact = (1 / (distances + 1))
    required_style_impact = required_style_impact.pow(1 / object_styles_impact_on_bg)
    plt.title("Heatmap for Object_Style1 Coefficients")
    hm = sns.heatmap(required_style_impact[:, :, 0].cpu().detach().numpy())
    plt.show()
    hm.get_figure().savefig(os.path.join(masks_directory, "obj1style_heatmap.jpg"))
    hm = sns.heatmap(required_style_impact[:, :, 1].cpu().detach().numpy())
    plt.title("Heatmap for Object_Style2 Coefficients")
    plt.show()
    hm.get_figure().savefig(os.path.join(masks_directory, "obj2style_heatmap.jpg"))
    required_style_impact = required_style_impact.unsqueeze(0).unsqueeze(0)
    return required_style_impact


def run_style_transfer(cnn,
                       normalization_mean,
                       normalization_std,
                       content_img,
                       style_img,
                       style_img2,
                       background_style_img,
                       input_img,
                       num_steps,
                       style_weight1,
                       style_weight2,
                       bg_style_weight,
                       content_weight,
                       multi_style_factor,
                       object_styles_impact_on_bg,
                       unloader,
                       stylized_output_name,
                       lr,
                       output_directory,
                       masks_directory):
    """Run the style transfer."""
    coefficients_for_object_styles = \
        calculate_pixel_distances_from_objects_and_convert_to_weights(content_img.clone(),
                                                                      unloader, masks_directory,
                                                                      object_styles_impact_on_bg)
    coefficients_for_background_style = (1 - torch.max(coefficients_for_object_styles, dim=-1)[0])
    plt.title("Heatmap for Background Style Coefficients")
    hm = sns.heatmap(coefficients_for_background_style[0][0].cpu().detach().numpy())
    hm.get_figure().savefig(os.path.join(masks_directory, "background_style_heatmap.jpg"))
    plt.show()
    input_img.requires_grad_(True)
    optimizer = get_input_optimizer(input_img, lr)
    iteration = [0]
    prev_total_loss = [float('inf')]
    curr_total_loss = [float('inf')]
    model, style_losses, style_losses2, background_style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                                                             normalization_mean,
                                                                                                             normalization_std,
                                                                                                             style_img,
                                                                                                             style_img2,
                                                                                                             background_style_img,
                                                                                                             content_img)
    model.requires_grad_(False)
    print("started optimization")
    while iteration[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            model(input_img)
            style_score = 0
            style_score2 = 0
            background_style_score = 0
            content_score = 0
            # seperate style and content loss calculations
            for sl in style_losses:
                style_score += sl.loss
            for sl2 in style_losses2:
                style_score2 += sl2.loss
            for sl3 in background_style_losses:
                background_style_score += sl3.loss
            for cl in content_losses:
                content_score += cl.loss
            background_style_score *= bg_style_weight
            style_score *= style_weight1
            style_score2 *= style_weight2
            content_score *= content_weight
            curr_total_loss[0] = content_score + style_score + style_score2 + background_style_score
            # linear blend of losses
            mixed_background_style_score = (1 - multi_style_factor) * background_style_score + multi_style_factor * (
                        style_score + style_score2)
            mixed_style_score = (1 - multi_style_factor) * style_score + multi_style_factor * background_style_score
            mixed_style_score2 = (1 - multi_style_factor) * style_score2 + multi_style_factor * background_style_score
            # seperate gradient derivations for each (blended) loss
            optimizer.zero_grad()
            mixed_background_style_score.backward(retain_graph=True)
            background_style_grad = input_img.grad.clone()
            optimizer.zero_grad()
            content_score.backward(retain_graph=True)
            content_grad = input_img.grad.clone()
            optimizer.zero_grad()
            mixed_style_score.backward(retain_graph=True)
            style1_grad = input_img.grad.clone()
            optimizer.zero_grad()
            mixed_style_score2.backward()
            style2_grad = input_img.grad.clone()
            # gradient mixer
            style_grad_combined = torch.concat((style1_grad.unsqueeze(-1), style2_grad.unsqueeze(-1)), dim=-1)
            style_grad_final = coefficients_for_object_styles * style_grad_combined
            style_grad_final_summed = torch.sum(style_grad_final, dim=-1)
            background_style_grad_final = coefficients_for_background_style * background_style_grad
            input_img.grad = content_grad + style_grad_final_summed + background_style_grad_final
            iteration[0] += 1
            # logs
            if iteration[0] % 50 == 0:
                print("iteration {}:".format(iteration))
                print("Total Loss: {:4f}".format(curr_total_loss[0].item()))
                print('Style Loss1 : {:4f} Style Loss2 : {:4f} Background Style Loss: {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), style_score2.item(), background_style_score.item(), content_score.item()))
                imshow(input_img, unloader)
                save_output(input_img, unloader, stylized_output_name)
            return curr_total_loss[0]

        # early stopping
        if curr_total_loss[0] <= prev_total_loss[0] or iteration[0] < 500:
            prev_total_loss[0] = curr_total_loss[0]
            optimizer.step(closure)
        else:
            break
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img


#   this is the basic framework. all boxes from onwards are abstractions on
#     top of it, for demo/submission purposes.
#   untirivial variables:
#     style_image_name = first object's style_reference
#     style_image2_name = second object's style reference
#     background_style_img_name = background's style reference
#     multi_style_factor - should be between 0.0 (distinct style transfer for each region -
#       each style loss will be a factor of itself alone) to 0.5 (multi-style transfer) -
#       small differences in styles between objects and background
#     object_styles_impact_on_bg: determines the extent to which the style of
#       the object should impact the style of the close bg pixels.
#       should be between 2 to 15
def preprocessing_and_nst(style_image_name, style_image2_name, background_style_img_name,
                          content_image_name, start_from_white_noise, num_steps,
                          style_weight1, style_weight2, bg_style_weight, content_weight,
                          multi_style_factor, object_styles_impact_on_bg, lr,
                          output_directory, masks_directory):
    # scale the image and transform to torch tensor
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    style_img, style_img2, background_style_img, content_img = image_loader(style_image_name, style_image2_name,
                                                                            background_style_img_name,
                                                                            content_image_name, loader)
    # reconvert into PIL image
    unloader = transforms.ToPILImage()
    plt.ion()
    cnn = models.vgg19(VGG19_Weights.DEFAULT).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    input_img = None
    if start_from_white_noise:
        input_img = torch.randn(content_img.data.size(), device=device)
    else:
        input_img = content_img.clone()
    plt.figure()
    stylized_output_name = os.path.join(output_directory, "stylized_output.jpg")
    imshow(input_img, unloader, title='Input Image')
    output = run_style_transfer(cnn, cnn_normalization_mean,
                                cnn_normalization_std,
                                content_img, style_img, style_img2,
                                background_style_img, input_img,
                                num_steps=num_steps,
                                style_weight1=style_weight1,
                                style_weight2=style_weight2,
                                bg_style_weight=bg_style_weight,
                                content_weight=content_weight,
                                multi_style_factor=multi_style_factor,
                                object_styles_impact_on_bg=object_styles_impact_on_bg,
                                unloader=unloader,
                                stylized_output_name=stylized_output_name,
                                lr=lr,
                                output_directory=output_directory,
                                masks_directory=masks_directory)

    plt.figure()
    imshow(style_img, unloader, title='Object1 style reference')
    plt.figure()
    imshow(style_img2, unloader, title='Object2 style reference')
    plt.figure()
    imshow(background_style_img, unloader, title='Background style reference')
    plt.figure()
    imshow(content_img, unloader, title='Content Image')
    plt.figure()
    imshow(output, unloader, title='Output Image')
    plt.ioff()
    plt.show()


# stylize objects
def flavor_one(content_image_name, style_image, num_steps,
               style_weight, bg_style_weight, content_weight,
               multi_style_factor,
               object_styles_impact_on_bg, outputs_dir_name, masks_dir_name):
    preprocessing_and_nst(style_image_name=style_image,
                          style_image2_name=style_image,
                          background_style_img_name=content_image_name,
                          content_image_name=content_image_name,
                          start_from_white_noise=False,
                          num_steps=num_steps,
                          style_weight1=style_weight,
                          style_weight2=style_weight,
                          bg_style_weight=bg_style_weight,
                          content_weight=content_weight,
                          multi_style_factor=multi_style_factor,
                          object_styles_impact_on_bg=object_styles_impact_on_bg,
                          lr=0.1,
                          output_directory=outputs_dir_name,
                          masks_directory=masks_dir_name)


# stylize background
def flavor_two(content_image_name, background_style_img_name, num_steps,
               style_weight, bg_style_weight, content_weight,
               multi_style_factor,
               object_styles_impact_on_bg,
               outputs_dir_name,
               masks_dir_name):
    preprocessing_and_nst(style_image_name=content_image_name,
                          style_image2_name=content_image_name,
                          background_style_img_name=background_style_img_name,
                          content_image_name=content_image_name,
                          start_from_white_noise=False,
                          num_steps=num_steps,
                          style_weight1=style_weight,
                          style_weight2=style_weight,
                          bg_style_weight=bg_style_weight,
                          content_weight=content_weight,
                          multi_style_factor=multi_style_factor,
                          object_styles_impact_on_bg=object_styles_impact_on_bg,
                          lr=0.1,
                          output_directory=outputs_dir_name,
                          masks_directory=masks_dir_name)


# stylize objects & background
def flavor_three(content_image_name, style_image_name, background_style_img_name,
                 num_steps, style_weight, bg_style_weight, content_weight,
                 multi_style_factor, object_styles_impact_on_bg,
                 outputs_dir_name, masks_dir_name):
    preprocessing_and_nst(style_image_name=style_image_name,
                          style_image2_name=style_image_name,
                          background_style_img_name=background_style_img_name,
                          content_image_name=content_image_name,
                          start_from_white_noise=False,
                          num_steps=num_steps,
                          style_weight1=style_weight,
                          style_weight2=style_weight,
                          bg_style_weight=bg_style_weight,
                          content_weight=content_weight,
                          multi_style_factor=multi_style_factor,
                          object_styles_impact_on_bg=object_styles_impact_on_bg,
                          lr=0.1,
                          output_directory=outputs_dir_name,
                          masks_directory=masks_dir_name)


def example_run_flavor_one():
    content_image_name = r"pradnyal-gandhi-9sCH94ihXSo-unsplash.jpg"
    style_image = r"f97c64edcbd4efc423b467960be76f53.jpg"
    outputs_dir_name = "outputs_flavor_one"
    masks_dir_name = "masks_and_heatmaps_flavor_one"
    if not os.path.isdir(outputs_dir_name):
        os.mkdir(outputs_dir_name)
    if not os.path.isdir(masks_dir_name):
        os.mkdir(masks_dir_name)
    flavor_one(content_image_name=content_image_name,
               style_image=style_image,
               num_steps=250,
               style_weight=1e7,
               bg_style_weight=1e6,
               content_weight=1e3,
               multi_style_factor=0.15,
               object_styles_impact_on_bg=2,
               outputs_dir_name=outputs_dir_name,
               masks_dir_name=masks_dir_name)


def example_run_flavor_two():
    content_image_name = r"content6.jpg"
    background_style_img_name = r"52.1524-4x3.jpg"
    outputs_dir_name = "outputs_flavor_two"
    masks_dir_name = "masks_and_heatmaps_flavor_two"
    if not os.path.isdir(outputs_dir_name):
        os.mkdir(outputs_dir_name)
    if not os.path.isdir(masks_dir_name):
        os.mkdir(masks_dir_name)
    flavor_two(content_image_name=content_image_name,
               background_style_img_name=background_style_img_name,
               num_steps=400,
               style_weight=1e6,
               bg_style_weight=5e7,
               content_weight=1e3,
               multi_style_factor=0.2,
               object_styles_impact_on_bg=1,
               outputs_dir_name=outputs_dir_name,
               masks_dir_name=masks_dir_name)


def example_run_flavor_three():
    content_image_name = r"content5.jpg"
    style_image_name = r"Jap-011863-George-Ward.jpg"
    background_style_img_name = r"claude-monet-houses-of-parliament.jpg"
    outputs_dir_name = "outputs_flavor_three"
    masks_dir_name = "masks_and_heatmaps_flavor_three"
    if not os.path.isdir(outputs_dir_name):
        os.mkdir(outputs_dir_name)
    if not os.path.isdir(masks_dir_name):
        os.mkdir(masks_dir_name)
    flavor_three(content_image_name=content_image_name,
                 style_image_name=style_image_name,
                 background_style_img_name=background_style_img_name,
                 num_steps=600,
                 style_weight=1e6,
                 bg_style_weight=1e8,
                 content_weight=1e2,
                 multi_style_factor=0.05,
                 object_styles_impact_on_bg=4,
                 outputs_dir_name=outputs_dir_name,
                 masks_dir_name=masks_dir_name)



# running instructions in colab:
  # 1 - drag the images (themselves, not the folder their in) from the
  # submission zip file to the filesystem to the
  # left of this box. After running the model, the outputs will
  #  also be available there.
  # 2 - run the various flavors (one at the time)
  # between runs it is best to perform: Runtime (in the menu on top) ->
  # Restart and run all, so the GPU memory will be cleared.
  # each run should take 2-4 minutes to completion (on colab's GPU)

# running instructions locally: (should run on a machine with GPU resources, else runtime would be long):
# drag the images (themselves, not the folder their in) to the directory of this main.py file
# run the various flavors, one at the time


example_run_flavor_one()
# example_run_flavor_two()
# example_run_flavor_three()

