import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

def adjust_gamma(image, gamma=None):
    if gamma == None:
        gamma = round(random.uniform(0.4,2),2)
        #print(gamma)
    to_PIL = transforms.ToPILImage()
    image = to_PIL(image)
    return TF.adjust_gamma(image, gamma)

def gaussian_blur(image, sigma=None, kernel_size=None):
    if kernel_size == None:
        kernel_size= 15
    if sigma == None:
        sigma = round(random.uniform(0.1,3),2)
        #print(sigma)
    to_PIL = transforms.ToPILImage()
    image = to_PIL(image)
    return TF.gaussian_blur(image, kernel_size, sigma)

def adjust_contrast(image, contrast_factor=None):
    if contrast_factor == None:
        contrast_factor = round(random.uniform(0.1,3),2)
        #print(contrast_factor)
    to_PIL = transforms.ToPILImage()
    image = to_PIL(image)
    return TF.adjust_contrast(image, contrast_factor)

def adjust_brightness(image, brightness_factor=None):
    if brightness_factor == None:
        brightness_factor = round(random.uniform(0.2,3),2)
        #print(brightness_factor)
    to_PIL = transforms.ToPILImage()
    image = to_PIL(image)
    return TF.adjust_brightness(image, brightness_factor)

def return_original(image):
    return image

def augmentation_policy(image, policy = None):
    #create augmentation policy, with same probabilty for each augmentation, choose one of them
    if policy == None:
        policy = random.randint(0,3)
    
    AUG_MAP = {0: return_original, 1: adjust_brightness, 2: adjust_gamma, 3: adjust_contrast, 4: gaussian_blur}
    
    return AUG_MAP[policy](image)

