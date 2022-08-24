#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
from .transform_image import *

class RandomCrop_P(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        we = im_lb['we']
        assert im.size == lb.size
        assert im.size == we.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb, we=we)
        if w < W or h < H: #expand
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)#int
            we = we.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop),
                we = we.crop(crop),
                    )


class HorizontalFlip_P(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            we = im_lb['we']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                        we = we.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class RandomScale_P(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        we = im_lb['we']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                    we = we.resize((w, h), Image.NEAREST),
                )


class ColorJitter_P(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]#[0.5, 1.5]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]#[0.5, 1.5]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]#[0.5, 1.5]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        we = im_lb['we']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im = im,
                    lb = lb,
                    we = we,
                )


class Compose_P(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


class Shadow_P(object):
    def __init__(self,shadow=(0.01, 0.3),shadow_file="shadow_pattern.jpg",shadow_crop_range=(0.02, 0.5) ):
        assert shadow[0] < shadow[1], "shadow max should be greater than shadow min"
        shadow_image = PIL.Image.open(shadow_file)
        # Ensure shadow is same color mode as input images
        self.shadow=shadow_image.convert("RGB")
        self.crop_range=shadow_crop_range
        self.intensity=shadow



    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        we = im_lb['we']
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
        return dict(im = im,
                    lb = lb,
                    we = we,
                )


class Noise_P(object):
    def __init__(self, noise=5, *args, **kwargs):
        self.noise_sd = np.random.randint(0, noise+1) #(0,5)

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        we = im_lb['we']
        if self.noise_sd > 0:
            noise = np.random.normal(loc=0, scale=self.noise_sd, size=np.shape(im))
            im2 = np.asarray(im, dtype=np.float32)  # prevent overflow
            im2 = np.clip(im2 + noise, 0, 255).astype(np.uint8)
            im= array2pil(im2)

        return dict(im = im,
                    lb = lb,
                    we = we,
                )

class Blur_P(object):
    def __init__(self, blur=0.3, *args, **kwargs):
        self.blur_radius = np.random.uniform(0, blur+1)

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        we = im_lb['we']
        if self.blur_radius > 0:
            im= im.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        return dict(im=im,
                    lb=lb,
                    we=we,
                    )


if __name__ == '__main__':
    flip = HorizontalFlip_P(p = 1)
    crop = RandomCrop_P((321, 321))
    rscales = RandomScale_P((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
