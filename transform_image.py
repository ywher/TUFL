#!/usr/bin/python
# -*- encoding: utf-8 -*-

import PIL
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
from PIL import ImageFilter,ImageOps
import random
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import torch
def random_90_rotation(im):
    """ Randomly rotates image in 90 degree increments
        (90, -90, or 180 degrees) """
    methods = [PIL.Image.ROTATE_90, PIL.Image.ROTATE_180, PIL.Image.ROTATE_270]
    method = np.random.choice(methods)
    return im.transpose(method=method)

def random_lr_flip(im):
    """ Randomly flips the image left-right with 0.5 probablility """
    if np.random.choice([0,1]) == 1:
        return im.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
    else:
        return im

def get_array_color_mode(x):
    """ Given a numpy array representing a single image, it returns the
        PIL color mode that will most likely work with it """
    x = x.squeeze()
    if x.ndim == 2:
        mode = "L"
    elif x.ndim == 3 and x.shape[2] == 1:
        mode = "L"
        x = x.squeeze()
    elif x.ndim == 3:
        mode = "RGB"
    else:
        assert False, "Incapable of interpreting array as an image"

    return mode

def random_tb_flip(im):
    """ Randomly flips the image top-bottom with 0.5 probablility """
    if np.random.choice([0,1]) == 1:
        return im.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)
    else:
        return im

def random_invert(im):
    """ With a 0.5 probability, it inverts the colors
        NOTE: This does not work on RGBA images yet. """
    assert im.mode != "RGBA", "Does random_invert not support RGBA images"
    if np.random.choice([0,1]) == 1:
        return PIL.ImageOps.invert(im)
    else:
        return im

def random_crop(im, min_scale=0.5, max_scale=1.0, preserve_size=False, resample=PIL.Image.NEAREST):
    """
    Args:
        im:         PIL image
        min_scale:   (float) minimum ratio along each dimension to crop from.
        max_scale:   (float) maximum ratio along each dimension to crop from.
        preserve_size: (bool) Should it resize back to original dims?
        resample:       resampling method during rescale.

    Returns:
        PIL image of size crop_size, randomly cropped from `im`.
    """
    assert (min_scale < max_scale), "min_scale MUST be smaller than max_scale"
    width, height = im.size
    crop_width = np.random.randint(width*min_scale, width*max_scale)
    crop_height = np.random.randint(height*min_scale, height*max_scale)

    # print('crop scale is ',min_scale)
    # crop_width =width*min_scale
    # crop_height=height*min_scale
    x_offset = np.random.randint(0, width - crop_width + 1)
    y_offset = np.random.randint(0, height - crop_height + 1)
    im2 = im.crop((x_offset, y_offset,
                   x_offset + crop_width,
                   y_offset + crop_height))
    if preserve_size:
        im2 = im2.resize(im.size, resample=resample)
    return im2

def is_numeric(x):
    return type(x) is int


def generate_random_lines(imshape,slant,drop_length,rain_type):
    drops=[]
    area=imshape[0]*imshape[1]
    no_of_drops=area//600

    if rain_type.lower()=="drizzle":
        no_of_drops=area//770
        drop_length=10
    elif rain_type.lower()=="heavy":
        drop_length=30
    elif rain_type.lower()=="torrential":
        no_of_drops=area//500
        drop_length=60

    for i in range(no_of_drops): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops,drop_length


def hls(image,src='RGB'):
    image_HLS = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HLS)')
    return image_HLS

def rgb(image, src='hls'):
    image_RGB= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)')
    return image_RGB

def rain_process(image,slant,drop_length,drop_color,drop_width,rain_drops):
    imshape = image.shape
    image_t= image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image= cv2.blur(image_t,(3,3)) ## rainy view are blurry
    brightness_coefficient = 0.7 ## rainy days are usually shady
    image_HLS = hls(image) ## Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB= rgb(image_HLS,'hls') ## Conversion to RGB
    return image_RGB


def array2pil(x):
    """ Given a numpy array containing image information returns a PIL image.
        Automatically handles mode, and even handles greyscale images with a
        channels axis
    """
    x = x.squeeze()
    return PIL.Image.fromarray(x, mode=get_array_color_mode(x))



class ColorJitter_image(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, sharpeness=None,*args, **kwargs):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.sharpeness = sharpeness


    def __call__(self,im):
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        r_sharpeness = random.uniform(self.sharpeness[0], self.sharpeness[1])

        # r_brightness= 1
        # r_contrast = 1
        # r_saturation = 1
        # r_sharpeness = 10

        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        im = ImageEnhance.Sharpness(im).enhance(r_sharpeness)
        return im

class Rain_image(object):
    def __init__(self,    rain='drizzle',rain_probablity=0.3 ):
        self.rain_probablity=rain_probablity
        self.rain_type=rain

    def __call__(self, image):
        slant=-1
        prob = np.random.uniform(0, 1)
        if prob < self.rain_probablity:
            err_rain_slant = "Numeric value between -20 and 20 is allowed"
            err_rain_width = "Width value between 1 and 5 is allowed"
            err_rain_length = "Length value between 0 and 100 is allowed"
            drop_width = 1
            # drop_length wiil be modified later
            drop_length = 10
            random_rain_color = np.random.randint(190, 200)
            drop_color = (random_rain_color, random_rain_color, random_rain_color)
            image = np.asarray(image)
            slant_extreme = slant
            if not (is_numeric(slant_extreme) and (
                    slant_extreme >= -20 and slant_extreme <= 20) or slant_extreme == -1):
                raise Exception(err_rain_slant)
            if not (is_numeric(drop_width) and drop_width >= 1 and drop_width <= 5):
                raise Exception(err_rain_width)
            if not (is_numeric(drop_length) and drop_length >= 0 and drop_length <= 100):
                raise Exception(err_rain_length)


            imshape = image.shape
            if slant_extreme == -1:
                slant = np.random.randint(-10, 10)  ##generate random slant if no slant value is given
            rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length, self.rain_type)
            im = rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops)
            image =array2pil(im)

        else:
            pass

        return image



class Shadow_image(object):
    def __init__(self,shadow=(0.01, 0.3),shadow_file="shadow_pattern.jpg",shadow_crop_range=(0.02, 0.5) ):
        assert shadow[0] < shadow[1], "shadow max should be greater than shadow min"
        shadow_image = PIL.Image.open(shadow_file)
        # Ensure shadow is same color mode as input images
        self.shadow=shadow_image.convert("RGB")
        self.crop_range=shadow_crop_range
        self.intensity=shadow



    def __call__(self, im):
        width, height = im.size
        assert im.mode == self.shadow.mode, "Scene image and shadow image must be same colorspace mode"

        # Take random crop from shadow image
        min_crop_scale, max_crop_scale = self.crop_range
        shadow = random_crop(self.shadow, min_scale=min_crop_scale, max_scale=max_crop_scale, preserve_size=False)
        shadow = shadow.resize((width, height), resample=PIL.Image.BILINEAR)

        # random flips, rotations, and color inversion
        shadow = random_tb_flip(random_lr_flip(random_90_rotation(shadow)))
        shadow = random_invert(shadow)
        # Ensure same shape as scene image after flips and rotations
        shadow = shadow.resize((width, height), resample=PIL.Image.BILINEAR)

        # Scale the shadow into proportional intensities (0-1)
        intensity_value = np.random.rand(1)
        min, max = self.intensity
        intensity_value = (intensity_value * (max - min)) + min  # remapped to min,max range
        # print('intensity is ',min)
        # intensity_value = min
        shadow = np.divide(shadow, 255)
        shadow = np.multiply(intensity_value, shadow)

        # Overlay the shadow
        overlay = (np.multiply(im, 1 - shadow)).astype(np.uint8)
        im=PIL.Image.fromarray(overlay, mode=im.mode)
        return im


class Noise_image(object):
    def __init__(self, noise=[0.5,1.5], *args, **kwargs):
        self.noise_sd = np.random.randint(noise[0], noise[1])

        # self.noise_sd= 12+1

    def __call__(self, im):
        if self.noise_sd > 0:
            noise = np.random.normal(loc=0, scale=self.noise_sd, size=np.shape(im))
            im2 = np.asarray(im, dtype=np.float32)  # prevent overflow
            im2 = np.clip(im2 + noise, 0, 255).astype(np.uint8)
            im= array2pil(im2)

        return im

class Blur_image(object):
    def __init__(self, blur=0.3, *args, **kwargs):
        self.blur_radius = np.random.uniform(0, blur+1)

    def __call__(self, im):
        if self.blur_radius > 0:
            im= im.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        return im


class Style_image(object):
    def __init__(self, p=0.5, *args, **kwargs):
        # self.choice=random.choice(['equalize','invert','solarize','posterize','greyscale'])
        # self.choice = random.choice(['greyscale','equalize'])#,'greyscale'
        self.choice = random.choice(['equalize'])  # ,'greyscale'
        # self.choice ='posterize'
        # print(self.choice)
        self.p=p

    def __call__(self, im):
        if random.random() < self.p:
            # print(self.choice)
    #return the image with the same grayscale image
            if self.choice=='equalize':
                im = ImageOps.equalize(im)
            elif self.choice=='greyscale':
                im = ImageOps.grayscale(im)
                im = im.convert('RGB')
    #invert all the pixels
            elif self.choice=='invert':
                im = ImageOps.invert(im)
   #inverth the pixel withing the threshold
            elif self.choice=='solarize':
                im = ImageOps.solarize(im,threshold=np.random.randint(100,150))
    #thresholding the bits to zero
            elif self.choice=='posterize':
                im = ImageOps.solarize(im,threshold=np.random.randint(3,6))

        return im


class RandomCrop_image(object):
    def __init__(self, size,*args, **kwargs):
        self.size = size
        self.random_factor=[random.random(),random.random()]

    def __call__(self, im):
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return im
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
        sw, sh = self.random_factor[0] * (w - W), self.random_factor[1] * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return im.crop(crop)


class RandomScale_image(object):
    def __init__(self, scales=(1), *args, **kwargs):
        self.scale = random.choice(scales)

    def __call__(self, im):
        W, H = im.size
        w, h = int(W * self.scale), int(H * self.scale)
        return im.resize((w, h), Image.BILINEAR)



class HorizontalFlip_image(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p
        self.random_seed=random.random()

    def __call__(self, im):
        if self.random_seed > self.p:
            return im
        else:
            return im.transpose(Image.FLIP_LEFT_RIGHT)



def Crop(im,crop_size,random_factor):
    W, H = crop_size
    w, h = im.size

    if (W, H) == (w, h): return dict(im=im)
    if w < W or h < H:
        scale = float(W) / w if w < h else float(H) / h
        w, h = int(scale * w + 1), int(scale * h + 1)
        im = im.resize((w, h), Image.BILINEAR)
    sw, sh = random_factor[0] * (w - W), random_factor[1]* (h - H)#random.random()
    crop = int(sw), int(sh), int(sw) + W, int(sh) + H
    return im.crop(crop)

def Crop_tensor(tensor,crop_size,random_factor):
    H,W = crop_size
    tensor_size_list=list(tensor.size())
    h,w  =tensor_size_list[-2],tensor_size_list[-1]
    # h,w  = tensor.size()[2],tensor.size()[3]

    if (W, H) == (w, h): return tensor
    if w < W or h < H:
        scale = float(W) / w if w < h else float(H) / h
        w, h = int(scale * w + 1), int(scale * h + 1)
        tensor = F.interpolate(tensor, (h, w), mode='bilinear', align_corners=True)
    sw, sh = random_factor[0] * (w - W), random_factor[1] * (h - H)  # random.random()
    crop = int(sh), int(sh) + H, int(sw), int(sw) + W
    if len(tensor_size_list)==3:
        tensor_cropped = tensor[:, crop[0]:crop[1], crop[2]:crop[3]]
    elif len(tensor_size_list)==4:
        tensor_cropped = tensor[:, :,crop[0]:crop[1], crop[2]:crop[3]]

    return tensor_cropped

def Scale_tensor(tensor,scale):
    H, W = list(tensor.size())[-2], list(tensor.size())[-1]
    w, h = int(W * scale), int(H * scale)
    # print("scale ",scale)
    # print("before ",H," ", W )
    # print("after ", h, " ", w)
    tensor = F.interpolate(tensor, (h, w), mode='bilinear', align_corners=True)
    return tensor

def HorizontalFlip_tensor(tensor,random_factor):
    if random_factor>0.5:
        tensor = tensor.flip(-1)

    return tensor

def affine_transform(tensor,crop_size,random_factor_crop,random_factor_scale,random_factor_hflip):
    # H, W = list(tensor.size())[-2], list(tensor.size())[-1]
    # tensor=HorizontalFlip_tensor(tensor,random_factor_hflip)
    # tensor = Crop_tensor(tensor, crop_size, random_factor_crop)
    # tensor=Scale_tensor(tensor,random_factor_scale)
    # tensor = F.interpolate(tensor, (H, W), mode='bilinear', align_corners=True)

    # random_factor_crop = [random.random(), random.random()]
    # random_factor_hflip = random.random()
    # random_factor_scale = random.choice((0.5, 0.75, 1.0, 1.25))
    # crop_size=[384,384]
    # if args.dataset == 'CityScapes':
    #     affine_crop_size = [384, 384]
    # elif args.dataset == 'Kitti_semantics':
    #     affine_crop_size = [300, 500]
    # im_unsup_aug = affine_transform(im_unsup_aug, affine_crop_size, random_factor_crop,
    #                                       random_factor_scale, random_factor_hflip)

    # print("im size ",im.size())
    # print("im_val size ", im_val.size())
    # print("im_unsup size ", im_unsup.size())
    # print("im_unsup_aug size ", im_unsup_aug.size())

    return tensor







#IIC PART
def random_affine(img, min_rot=None, max_rot=None, min_shear=None,
                  max_shear=None, min_scale=None, max_scale=None):
  # Takes and returns torch cuda tensors with channels 1st (1 img)
  # rot and shear params are in degrees
  # tf matrices need to be float32, returned as tensors
  # we don't do translations

  # https://github.com/pytorch/pytorch/issues/12362
  # https://stackoverflow.com/questions/42489310/matrix-inversion-3-3-python
  # -hard-coded-vs-numpy-linalg-inv

  # https://github.com/pytorch/vision/blob/master/torchvision/transforms
  # /functional.py#L623
  # RSS(a, scale, shear) = [cos(a) *scale   - sin(a + shear) * scale     0]
  #                        [ sin(a)*scale    cos(a + shear)*scale     0]
  #                        [     0                  0          1]
  # used by opencv functional _get_affine_matrix and
  # skimage.transform.AffineTransform

  assert (len(img.shape) == 3)
  random_angle=np.random.rand() * (max_rot - min_rot) + min_rot
  a = np.radians(random_angle)
  shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
  scale = np.random.rand() * (max_scale - min_scale) + min_scale

  affine1_to_2 = np.array([[np.cos(a) * scale, - np.sin(a + shear) * scale, 0.],
                           [np.sin(a) * scale, np.cos(a + shear) * scale, 0.],
                           [0., 0., 1.]], dtype=np.float32)  # 3x3

  affine2_to_1 = np.linalg.inv(affine1_to_2).astype(np.float32)

  affine1_to_2, affine2_to_1 = affine1_to_2[:2, :], affine2_to_1[:2, :]  # 2x3
  # affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2).cuda(), \
  #                              torch.from_numpy(affine2_to_1).cuda()

  affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2).cpu(),\
                               torch.from_numpy(affine2_to_1).cpu()

  if np.random.rand() > 0.5:
      # applied affine, then flip, new = flip * affine * coord
      # (flip * affine)^-1 is just flip^-1 * affine^-1.
      # No order swap, unlike functions...
      # hence top row is negated
      affine2_to_1[0, :] *= -1.
      affine1_to_2[0, :] *= -1.


  img=img.unsqueeze(dim=0).cpu()
  # print("img ",img.type())
  img = perform_affine_tf(img, affine1_to_2.unsqueeze(dim=0))
  img = img.squeeze(dim=0)

  return img, affine1_to_2, affine2_to_1

def fixed_affine(img, rot=None, shear=None, scale=None):
  # Takes and returns torch cuda tensors with channels 1st (1 img)
  # rot and shear params are in degrees
  # tf matrices need to be float32, returned as tensors
  # we don't do translations

  # https://github.com/pytorch/pytorch/issues/12362
  # https://stackoverflow.com/questions/42489310/matrix-inversion-3-3-python
  # -hard-coded-vs-numpy-linalg-inv

  # https://github.com/pytorch/vision/blob/master/torchvision/transforms
  # /functional.py#L623
  # RSS(a, scale, shear) = [cos(a) *scale   - sin(a + shear) * scale     0]
  #                        [ sin(a)*scale    cos(a + shear)*scale     0]
  #                        [     0                  0          1]
  # used by opencv functional _get_affine_matrix and
  # skimage.transform.AffineTransform

  assert (len(img.shape) == 3)
  angle=rot
  shear = np.radians(shear)
  a = np.radians(angle)

  affine1_to_2 = np.array([[np.cos(a) * scale, - np.sin(a + shear) * scale, 0.],
                           [np.sin(a) * scale, np.cos(a + shear) * scale, 0.],
                           [0., 0., 1.]], dtype=np.float32)  # 3x3

  affine2_to_1 = np.linalg.inv(affine1_to_2).astype(np.float32)

  affine1_to_2, affine2_to_1 = affine1_to_2[:2, :], affine2_to_1[:2, :]  # 2x3
  # affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2).cuda(), \
  #                              torch.from_numpy(affine2_to_1).cuda()

  affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2).cpu(),\
                               torch.from_numpy(affine2_to_1).cpu()

  img=img.unsqueeze(dim=0).cpu()
  # print("img ",img.type())
  img = perform_affine_tf(img, affine1_to_2.unsqueeze(dim=0))
  img = img.squeeze(dim=0)

  return img, affine1_to_2, affine2_to_1


def perform_affine_tf(data, tf_matrices):
  # expects 4D tensor, we preserve gradients if there are any
  # print("data ",data.type())
  n_i, k, h, w = data.shape
  n_i2, r, c = tf_matrices.shape
  assert (n_i == n_i2)
  assert (r == 2 and c == 3)

  grid = F.affine_grid(tf_matrices, data.shape)  # output should be same size


  data_tf = F.grid_sample(data, grid,
                          padding_mode="zeros")  # this can ONLY do bilinear

  return data_tf


def random_translation_multiple(data, half_side_min, half_side_max):
  n, c, h, w = data.shape

  # pad last 2, i.e. spatial, dimensions, equally in all directions
  data = F.pad(data,
               (half_side_max, half_side_max, half_side_max, half_side_max),
               "constant", 0)
  assert (data.shape[2:] == (2 * half_side_max + h, 2 * half_side_max + w))

  # random x, y displacement
  t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
  polarities = np.random.choice([-1, 1], size=(2,), replace=True)
  t *= polarities

  # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
  t += half_side_max

  data = data[:, :, t[1]:(t[1] + h), t[0]:(t[0] + w)]
  assert (data.shape[2:] == (h, w))

  return data


def random_translation(img, half_side_min, half_side_max):
  # expects 3d (cuda) tensor with channels first
  c, h, w = img.shape

  # pad last 2, i.e. spatial, dimensions, equally in all directions
  img = F.pad(img, (half_side_max, half_side_max, half_side_max, half_side_max),
              "constant", 0)
  assert (img.shape[1:] == (2 * half_side_max + h, 2 * half_side_max + w))

  # random x, y displacement
  t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
  polarities = np.random.choice([-1, 1], size=(2,), replace=True)
  t *= polarities

  # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
  t += half_side_max

  img = img[:, t[1]:(t[1] + h), t[0]:(t[0] + w)]
  assert (img.shape[1:] == (h, w))

  return img


if __name__ == '__main__':
    # flip = HorizontalFlip_image(p = 1)
    # crop = RandomCrop_image((321, 321))
    # rscales = RandomScale_image((0.75, 1.0, 1.5, 1.75, 2.0))
    # scale = random.choice(scales)
    img = Image.open('../data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png')
    # img.show()
    random_factor=[random.random(),random.random()]
    crop_size=[512,512]
    img_crop=Crop(img,crop_size,random_factor)
    img_crop.show()
    trans1 = transforms.ToTensor()
    tensor=trans1(img)
    tensor=Crop_tensor(tensor, crop_size, random_factor)
    tensor=tensor.flip(-1)
    trans2=transforms.ToPILImage()
    img_tensor_crop=trans2(tensor)
    img_tensor_crop.show()


