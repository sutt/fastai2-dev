import traceback
from fastai2.vision.all import *
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")

path = Path('../../rf-chess-data/cropped_v1/')
fns = get_image_files(path)  # works recursively, to each subfolder

# failed = verify_images(fns)

print(fns[:3])

def piece_class_parse(fn): 
    fn = fn.split('_')[1]
    fn = fn.split('.')[0]
    return fn

pieces = ImageDataLoaders.from_name_func(
                path, 
                get_image_files(path),
                valid_pct=0.2, 
                seed=42,
                label_func=piece_class_parse, 
#                 item_tfms=Resize(128),
                item_tfms=RandomResizedCrop(128, min_scale=0.5),
                batch_tfms=aug_transforms(),
                )


catchme = pieces.train.show_batch(max_n=8, nrows=1, unique=True)


print('endd')