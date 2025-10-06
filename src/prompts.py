
def get_templates(dataset):
    dataset_templates = {}
    dataset_templates['cifar10'] = get_cifar10_templates()
    dataset_templates['cifar10animals'] = get_cifar10_templates()
    dataset_templates['cifar10objects'] = get_cifar10_templates()
    dataset_templates['cifar100'] = get_cifar100_templates()
    dataset_templates['food101'] = get_food101_templates()
    dataset_templates['country211'] = get_country211_templates()
    dataset_templates['mnist'] = get_mnist_templates()
    dataset_templates['fmnist'] = get_fmnist_templates()
    dataset_templates['oxfordpets'] = get_cifar10_templates()
    dataset_templates['imagenet-a'] = get_imagenet_templates()
    dataset_templates['imagenet-r'] = get_imagenet_templates()
    dataset_templates['fairface-class'] = get_fairface_templates()
    dataset_templates['fairface-skew'] = get_occupation_templates()
    dataset_templates['socialcounterfactuals-class'] = get_fairface_templates()
    dataset_templates['socialcounterfactuals-skew'] = get_occupation_templates()

    return dataset_templates[dataset]

def get_cifar10_templates():
    templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
    ]

    return templates

def get_cifar100_templates():
    templates = [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.',
        'a blurry photo of the {}.',
        'a black and white photo of the {}.',
        'a low contrast photo of the {}.',
        'a high contrast photo of the {}.',
        'a bad photo of the {}.',
        'a good photo of the {}.',
        'a photo of the small {}.',
        'a photo of the big {}.',
    ]

    return templates

def get_food101_templates():
    templates = [
    'a photo of {}, a type of food.',
    ]

    return templates

def get_country211_templates():
    templates = [
    'a photo i took in {}.',
    'a photo i took while visiting {}.',
    'a photo from my home country of {}.',
    'a photo from my visit to {}.',
    'a photo showing the country of {}.',
    ]

    return templates

def get_mnist_templates():
    templates = [
    'a photo of the number: "{}".',
    ]

    return templates

def get_fmnist_templates():
    templates = [
    'a photo of a "{}".',
    ]

    return templates

def get_imagenet_templates():
    templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
    ]

    return templates

def get_occupation_templates():
    templates = ['A <occupation>', 'A photo of <occupation>',
                'A picture of <occupation>', 'An image of <occupation>']

    return templates

def get_fairface_templates():
    templates = ['An image of a {} person',
                'A portrait of a {} person',
                'An image of a {} person',
                'A face of a {} person',
                'A headshot of a {} individual',
                'A close-up photo of a {} person']
    
    return templates
