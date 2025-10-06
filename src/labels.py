import os

def get_data_paths(dataset):
    datasets = {}
    datasets['cifar10'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/datasets/CIFAR-10-images/'
    datasets['cifar10animals'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/datasets/CIFAR-10-images/'
    datasets['cifar10objects'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/datasets/CIFAR-10-images/'
    datasets['food101'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/datasets/food-101/'
    datasets['cifar100'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/datasets/CIFAR-100-dataset/'
    datasets['mnist'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/datasets/mnist_png/'
    datasets['country211'] = '/export/share/projects/mcai/datasets/country211/'
    datasets['inaturalist'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/datasets/iNaturalist/'
    datasets['fmnist'] = '/export/share/projects/mcai/datasets/fashion-mnist/'
    datasets['oxfordpets'] = '/export/share/projects/mcai/datasets/oxford-pets/'
    datasets['imagenet-a'] = '/export/share/projects/mcai/datasets/imagenet-a/'
    datasets['imagenet-r'] = '/export/share/projects/mcai/datasets/imagenet-r/'
    datasets['kinetics700-act-rec'] = '/home/amadasu/Experiments/data/kinetics-700-val-trimmed/',
    datasets['kinetics600-act-rec'] = '/home/amadasu/Experiments/data/kinetics-700-val-trimmed/',
    datasets['ucf-act-rec'] =  '/home/amadasu/Experiments/data/ucf101_videos/'
    datasets['fairface-class'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/avinash/datasets/fairface/'
    datasets['fairface-skew'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/avinash/datasets/fairface/'
    datasets['socialcounterfactuals-class'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/counterfactuals/mm_bias/final_dataset/'
    datasets['socialcounterfactuals-skew'] = '/mnt/beegfs/mixed-tier/share/projects/mcai/counterfactuals/mm_bias/final_dataset/'

    return datasets[dataset]

def get_occupations(dataset):
    
    keywords= ['biologist', 'composer', 'economist', 'mathematician', 'model', 'poet', 'reporter', 'zoologist', 
            'artist', 'coach', 'athlete', 'audiologist', 'judge', 'musician', 'therapist', 'banker', 'ceo', 
            'consultant', 'prisoner', 'assistant', 'boxer', 'commander', 'librarian', 'nutritionist', 
            'realtor', 'supervisor', 'architect', 'priest', 'guard', 'magician', 'producer', 'teacher', 
            'lawyer', 'paramedic', 'researcher', 'physicist', 'pediatrician', 'surveyor', 'laborer', 
            'statistician', 'dietitian', 'sailor', 'tailor', 'attorney', 'army', 'manager', 'baker', 
            'recruiter', 'clerk', 'entrepreneur', 'sheriff', 'policeman', 'businessperson', 'chief', 
            'scientist', 'carpenter', 'florist', 'optician', 'salesperson', 'umpire', 'painter', 'guitarist', 
            'broker', 'pensioner', 'soldier', 'astronaut', 'dj', 'driver', 'engineer', 'cleaner', 'cook', 
            'housekeeper', 'swimmer', 'janitor', 'pilot', 'mover', 'handyman', 'firefighter', 'accountant', 
            'physician', 'farmer', 'bricklayer', 'photographer', 'surgeon', 'dentist', 'pianist', 'hairdresser', 
            'receptionist', 'waiter', 'butcher', 'videographer', 'cashier', 'technician', 'chemist', 
            'blacksmith', 'dancer', 'doctor', 'nurse', 'mechanic', 'chef', 'plumber', 'bartender', 
            'pharmacist', 'electrician']
    
    '''
    else:
        keywords = ['composer', 'mathematician', 'model', 'poet', 'reporter', 'athlete', 'economist', 'zoologist', 'coach', 
                    'prisoner', 'commander', 'chief', 'army', 'banker', 'magician', 'supervisor', 'judge', 'musician', 
                    'swimmer', 'priest', 'umpire', 'laborer', 'nutritionist', 'boxer', 'architect', 'librarian', 
                    'producer', 'surveyor', 'audiologist', 'consultant', 'physicist', 'soldier', 'tailor', 'therapist', 
                    'statistician', 'sailor', 'researcher', 'policeman', 'manager', 'attorney', 'clerk', 'mover', 
                    'guard', 'sheriff', 'lawyer', 'paramedic', 'ceo', 'dietitian', 'artist', 'pediatrician']
    '''
    return keywords

def get_cifar10_labels(is_joined):
    classes = ['airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if(is_joined):
        classes = ["_".join(classname.split()) for classname in classes]
    return classes

def get_cifar10animals_labels(is_joined):
    classes = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']

    if(is_joined):
        classes = ["_".join(classname.split()) for classname in classes]
    return classes

def get_cifar10objects_labels(is_joined):
    classes = ['airplane', 'automobile', 'ship', 'truck']

    if(is_joined):
        classes = ["_".join(classname.split()) for classname in classes]
    return classes

def get_food101_labels(is_joined=True):
    classes = ['apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare',
            'beet salad', 'beignets', 'bibimbap', 'bread pudding', 'breakfast burrito',
            'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake',
            'ceviche', 'cheese plate', 'cheesecake', 'chicken curry', 'chicken quesadilla',
            'chicken wings', 'chocolate cake', 'chocolate mousse', 'churros', 'clam chowder',
            'club sandwich', 'crab cakes', 'creme brulee', 'croque madame', 'cup cakes',
            'deviled eggs', 'donuts', 'dumplings', 'edamame', 'eggs benedict', 'escargots',
            'falafel', 'filet mignon', 'fish and chips', 'foie gras', 'french fries',
            'french onion soup', 'french toast', 'fried calamari', 'fried rice', 'frozen yogurt',
            'garlic bread', 'gnocchi', 'greek salad', 'grilled cheese sandwich', 'grilled salmon',
            'guacamole', 'gyoza', 'hamburger', 'hot and sour soup', 'hot dog', 'huevos rancheros',
            'hummus', 'ice cream', 'lasagna', 'lobster bisque', 'lobster roll sandwich',
            'macaroni and cheese', 'macarons', 'miso soup', 'mussels', 'nachos', 'omelette',
            'onion rings', 'oysters', 'pad thai', 'paella', 'pancakes', 'panna cotta',
            'peking duck', 'pho', 'pizza', 'pork chop', 'poutine', 'prime rib',
            'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake', 'risotto', 'samosa',
            'sashimi', 'scallops', 'seaweed salad', 'shrimp and grits', 'spaghetti bolognese',
            'spaghetti carbonara', 'spring rolls', 'steak', 'strawberry shortcake', 'sushi',
            'tacos', 'takoyaki', 'tiramisu','tuna tartare', 'waffles']
    
    if(is_joined):
        classes = ["_".join(classname.split()) for classname in classes]
    return classes

def get_mnist_labels(is_joined):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return classes

def get_fmnist_labels(is_joined):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
            'Sneaker', 'Bag', 'Ankle boot']
    
    return classes

def get_inaturalist_labels(is_joined):
    classes = ['Plantae', 'Insecta', 'Aves', 'Reptilia', 'Mammalia', 'Fungi', 'Amphibia',
            'Mollusca', 'Animalia', 'Arachnida', 'Actinopterygii', 'Chromista', 'Protozoa']
    
    return classes

def get_cifar100_labels(is_joined=True):
    classes = [
        'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear',
        'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm'
    ]
    
    if(is_joined):
        classes = ["_".join(classname.split()) for classname in classes]
    return classes

def get_country211_labels(is_joined):
    classes = ['India', 'Guam', 'Montenegro', "Cote d'Ivoire", 'Brazil', 'Gibraltar', 'Luxembourg', 'Papua New Guinea', 
               'New Caledonia', 'Tajikistan', 'Ethiopia', 'Mozambique', 'Belarus', 'Saint Martin (French part)', 'Panama', 
               'Puerto Rico', 'DR Congo', 'Qatar', 'Saint Helena, Ascension and Tristan da Cunha', 'Bhutan', 'Cayman Islands', 
               'Denmark', 'Egypt', 'Nepal', 'Macedonia, the Former Yugoslav Republic of', 'Aruba', 'Morocco', 'Sudan', 'Poland', 
               'Dominica', 'Zimbabwe', 'Dominican Republic', 'Martinique', 'Liechtenstein', 'Singapore', 'Fiji', 'Iceland', 
               'Malawi', 'Switzerland', 'French Guiana', 'Peru', 'Benin', 'Hungary', 'Saudi Arabia', 'Mexico', 'Iran', 'Liberia', 
               'Algeria', 'Nicaragua', 'Ghana', 'Yemen', 'Sint Maarten (Dutch part)', 'Germany', 'Bangladesh', 'Andorra', 'Honduras', 
               'Paraguay', 'Libya', 'Turkmenistan', 'Sri Lanka', 'Guadeloupe', 'Gambia', 'French Polynesia', 'Viet Nam', 'Chile', 
               'South Sudan', 'Georgia', 'China', 'Cook Islands', 'Romania', 'Mongolia', 'Oman', 'Bahamas', 'Malta', 'Austria', 
               'Guernsey', 'Palestine, State of', 'United Kingdom', 'Slovakia', 'Uzbekistan', 'Jamaica', 'Cambodia', 'Tonga', 
               'Kuwait', 'Portugal', 'Ecuador', 'Ireland', 'Bolivia', 'Finland', 'Canada', 'Uganda', 'Belgium', 
               'Taiwan, Province of China', 'Antarctica', 'Colombia', 'Faeroe Islands', 'Cuba', 'Serbia', 'Gabon', 'Pakistan', 
               'Sierra Leone', 'Ukraine', 'Israel', 'Bermuda', 'Mauritius', 'Falkland Islands', 'Cabo Verde', 'South Korea', 
               'Jordan', 'Syrian Arab Republic', 'Bahrain', 'Tanzania, United Republic of', 'Virgin Islands, British', 
               'Somalia', 'Guyana', 'Afghanistan', 'South Georgia and South Sandwich Is.', 'Macao', 'Angola', 'South Africa', 
               'Albania', 'North Korea', 'Solomon Islands', 'Maldives', 'Estonia', 'Holy See (Vatican City State)', 'Curacao', 
               'Slovenia', 'Palau', 'Nigeria', 'Mali', 'Isle of Man', 'Lithuania', 'Barbados', 'Croatia', 'Malaysia', 'Laos', 
               'Russian Federation', 'Myanmar', 'Kenya', 'Madagascar', 'Aland Islands', 'Saint Kitts and Nevis', 'Tunisia', 
               'Bulgaria', 'Namibia', 'Eswatini', 'Senegal', 'Botswana', 'Venezuela, Bolivarian Republic of', 'Kazakhstan', 
               'Costa Rica', 'Zambia', 'Hong Kong', 'Virgin Islands, U.S.', 'Monaco', 'Trinidad and Tobago', 'Australia', 
               'Mauritania', 'Italy', 'El Salvador', 'Argentina', 'France', 'San Marino', 'Norway', 'Brunei Darussalam', 
               'Czech Republic', 'Armenia', 'Seychelles', 'Indonesia', 'Bosnia and Herzegovina', 'Antigua and Barbuda', 
               'Cameroon', 'Bonaire, Saint Eustatius and Saba', 'Burkina Faso', 'Kyrgyzstan', 'Central African Republic', 
               'Japan', 'Guatemala', 'Rwanda', 'Netherlands', 'Azerbaijan', 'Belize', 'Timor-Leste', 'Philippines', 
               'Iraq', 'RÃ©union', 'Haiti', 'Thailand', 'Kosovo', 'New Zealand', 'Uruguay', 'Vanuatu', 'Togo', 
               'Sweden', 'Turkey', 'Lebanon', 'Anguilla', 'Moldova, Republic of', 'Samoa', 'Svalbard and Jan Mayen', 
               'Greenland', 'Greece', 'Spain', 'United States', 'Grenada', 'United Arab Emirates', 'Jersey', 'Cyprus', 
               'St. Lucia', 'Latvia']
    
    if(is_joined):
        classes = ["_".join(classname.split()) for classname in classes]
    return classes

def get_oxfordpets_labels(is_joined):
    classes = ['american', 'basset', 'beagle', 'boxer', 'chihuahua', 'english', 'german', 'great', 'havanese', 
                'japanese', 'keeshond', 'leonberger', 'miniature', 'newfoundland', 'pomeranian', 'pug', 'saint', 
                'samoyed', 'scottish', 'shiba', 'staffordshire', 'wheaten', 'yorkshire', 'Abyssinian', 'Bengal', 
                'Birman', 'Bombay', 'British', 'Egyptian', 'Maine', 'Persian', 'Ragdoll', 
                'Russian', 'Siamese', 'Sphynx']

    return classes

def get_imageneta_labels(is_joined):
    classes = []
    with open(os.path.join(get_data_paths('imagenet-a'), 'README.txt'), 'r') as f:
        for line in f.readlines():
            classes.append(" ".join(line.strip().split()[1:]))
    
    return classes

def get_imagenetr_labels(is_joined):
    classes = []
    with open(os.path.join(get_data_paths('imagenet-r'), 'README.txt'), 'r') as f:
        for line in f.readlines():
            curclass = line.strip().split()[1]
            classes.append(" ".join(curclass.split("_")))
    
    return classes

def get_fairface_labels(attribute, is_joined):
    if(attribute == 'race'):
        classes = ['Asian', 'Black', 'Indian', 'Latino', 'Middle Eastern', 'White']
    else:
        classes = ['Male', 'Female']
    
    return classes

def get_socialcounterfactuals_labels(attribute, is_joined):
    if(attribute == 'race'):
        classes = ['Asian', 'Black', 'Indian', 'Latino', 'Middle Eastern', 'White']
    else:
        classes = ['Male', 'Female']
    
    return classes
