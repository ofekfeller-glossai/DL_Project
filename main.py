from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import os
from scipy import ndimage


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
# imsize = 512 if torch.cuda.is_available() else 40  # use small size if no gpu


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
    return style_image.to(device, torch.float), style_image2.to(device, torch.float), background_style_image.to(device, torch.float), content_image.to(device, torch.float)


def imshow(tensor, unloader, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save("output.jpg")
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
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

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
        if name in style_layers:
            # add style loss:
            target_feature = model(style_img2).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style2_loss_{}".format(i), style_loss)
            style_losses2.append(style_loss)


        # background style
        if name in style_layers:
            # add style loss:
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


def save_output(input_img, unloader, filename):  # Maybe add clamping/clipping
    image = input_img.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(filename)


def calc_distances_from_mask(mask):
    flipped_mask = 1 - mask
    distances_from_mask = torch.tensor(ndimage.distance_transform_edt(flipped_mask.cpu()), requires_grad=False).to(device, torch.float)
    return distances_from_mask



def get_objects_mask(content_img):
  seg_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device).eval().requires_grad_(False)
  content_seg = seg_model(content_img)[0]
  boxes, masks = content_seg['boxes'], content_seg['masks']
  boxes_squares = torch.tensor([(boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]) for i in range(boxes.shape[0])])
  max_squares, largest_objects_indices = boxes_squares.topk(2)
  first_obj_mask, second_obj_mask = masks[largest_objects_indices[0]], masks[largest_objects_indices[1]]
  first_obj_mask[first_obj_mask > 0.5] = 1.0
  first_obj_mask[first_obj_mask <= 0.5] = 0.0
  second_obj_mask[second_obj_mask > 0.5] = 1.0
  second_obj_mask[second_obj_mask <= 0.5] = 0.0
  objects_img = content_img*(first_obj_mask+second_obj_mask)
  unloader = transforms.ToPILImage()  # reconvert into PIL image
  plt.figure()
  imshow(objects_img, unloader, title='segmented objects')
  stacked_mask = torch.concat((first_obj_mask.unsqueeze(-1), second_obj_mask.unsqueeze(-1)), dim=-1).squeeze(0)
  return stacked_mask


#   object_styles_impact_on_bg: determines the extent to which the style of 
#   the object should impact the style of the close bg pixels. 
#   should be between 2 to 6
def calculate_pixel_distances_from_objects(input_img, 
                                           object_styles_impact_on_bg=4):
    objects_mask = get_objects_mask(input_img)
    distances = calc_distances_from_mask(objects_mask)
    required_style_impact = (1 / (distances + 1)).pow(1 / object_styles_impact_on_bg)
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
                       style_weight, 
                       bg_style_weight,
                       content_weight, 
                       object_styles_impact_on_bg,
                       unloader, 
                       filename, 
                       lr):
  
    """Run the style transfer."""
    print('Building the style transfer model..')
    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    probs = calculate_pixel_distances_from_objects(content_img.clone(), object_styles_impact_on_bg)
    input_img.requires_grad_(True)
    optimizer = get_input_optimizer(input_img, lr)
    run = [0]
    min_loss = [float('inf')]
    model, style_losses, style_losses2, background_style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, 
                                                                     style_img,style_img2, background_style_img,
                                                                     content_img)
    model.requires_grad_(False)
    probs_for_background_style = (1 - torch.max(probs, dim=-1)[0])
    while run[0] <= num_steps:
        # style_weight *= 0.98  # TODO change to decaying probs
        def closure():
            with torch.no_grad():
              input_img.clamp_(0, 1)
            model(input_img)
            style_score = 0
            style_score2 = 0
            background_style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for sl2 in style_losses2:
                style_score2 += sl2.loss
            for sl3 in background_style_losses:
                background_style_score += sl3.loss
            for cl in content_losses:
                content_score += cl.loss
            background_style_score *= bg_style_weight
            style_score *= style_weight
            style_score2 *= style_weight
            content_score *= content_weight
            total_loss = content_score + style_score + style_score2 + background_style_score
            background_style_score += 0.01 * (style_score+style_score2+content_score)
            style_score += 0.1 * (background_style_score+content_score)
            style_score2 += 0.1 * (background_style_score+content_score)
            optimizer.zero_grad()
            background_style_score.backward(retain_graph=True)
            background_style_grad = input_img.grad.clone()
            optimizer.zero_grad()
            (content_score).backward(retain_graph=True)
            content_grad = input_img.grad.clone()
            optimizer.zero_grad()
            style_score.backward(retain_graph = True)
            style1_grad = input_img.grad.clone()
            optimizer.zero_grad()
            style_score2.backward()
            style2_grad = input_img.grad.clone()
            style_grad_combined = torch.concat((style1_grad.unsqueeze(-1), style2_grad.unsqueeze(-1)), dim=-1)
            style_grad_final = probs * style_grad_combined 
            style_grad_final_summed = torch.sum(style_grad_final, dim=-1)
            background_style_grad_final = probs_for_background_style * background_style_grad 
            input_img.grad = content_grad + style_grad_final_summed + background_style_grad_final
            run[0] += 1
            if run[0] % 50 == 0:
              print("run {}:".format(run))
              print("Total Loss: {:4f}".format(total_loss.item()))
              print('Style Loss1 : {:4f} Style Loss2 : {:4f} Background Style Loss: {:4f} Content Loss: {:4f}'.format(
                  style_score.item(), style_score2.item(), background_style_score.item(), content_score.item()))
              imshow(input_img, unloader)
              save_output(input_img, unloader, filename)
            return style_score + style_score2 + content_score
        optimizer.step(closure)
    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img


def nst_main(style_image_name, style_image2_name, background_style_img_name, 
             content_image_name, start_from_white_noise, num_steps,
             style_weight, bg_style_weight, content_weight, object_styles_impact_on_bg, lr):
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    style_img, style_img2, background_style_img, content_img = image_loader(style_image_name, style_image2_name, background_style_img_name, content_image_name, loader)
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    plt.ion()
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    input_img = None
    if start_from_white_noise:
        input_img = torch.randn(content_img.data.size(), device=device)
    else:
        input_img = content_img.clone()
    # add the original input image to the figure:
    plt.figure()
    filename = os.path.join("/content/outputs", "style-"+
                            style_image_name.split("/")[1].split(".")[0]+
                            "-content-"+content_image_name.split("/")[1].split(".")[0]+".jpg")
    imshow(input_img, unloader, title='Input Image')
    output = run_style_transfer(cnn, cnn_normalization_mean, 
                                cnn_normalization_std,
                                content_img, style_img, style_img2, 
                                background_style_img, input_img,
                                num_steps=num_steps,
                                style_weight=style_weight,
                                bg_style_weight=bg_style_weight,
                                content_weight=content_weight,
                                object_styles_impact_on_bg=object_styles_impact_on_bg,
                                unloader=unloader, 
                                filename=filename, 
                                lr=lr)

    plt.figure()
    imshow(style_img, unloader, title='Style Image')
    plt.figure()
    imshow(content_img, unloader, title='Content Image')
    plt.figure()
    imshow(output, unloader, title='Output Image')
    plt.ioff()
    plt.show()

    
content_image_name = r"content/beach.jpg"
style_image1 = r"style/van.jpg"
style_image2 = r"style/van.jpg"
background_style_img = r"content/beach.jpg"
nst_main(style_image_name=style_image1, 
         style_image2_name=style_image1, 
         background_style_img_name=content_image_name,
         content_image_name=content_image_name, 
         start_from_white_noise=False, 
         num_steps=1e4,
         style_weight=1e7, 
         bg_style_weight=1e6,
         content_weight=400, 
         object_styles_impact_on_bg=2,
         lr=0.1)
 
