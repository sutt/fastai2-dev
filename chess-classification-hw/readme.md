# Chess Classification Homework

  *Dataset compliments of [Roboflow](https://roboflow.ai/)* 

Building an image classification pipeline with chess pieces. The *general machine learning* question we will address here:
      
> How to build a classification model that transfers well to targets slightly different from train set? Or, what is the minimal effort to perform this transfer?

*Questions / Investigations:*

 - Can you predict piece class on white peices when only trained on black?

 - How well does a model trained on standard peices perform on slighly non standard pieces (referred to as "wooden pieces"?

 - What happens with piece occlusion?

 | Trainging Targets  | Test-Set Targets |
| --- | --- | 
|   *"regulation dataset"* |   *"wooden dataset"*|
|<img src="img-public/regulation-white-bishop-1.jpg" width="100px"/> <img src="img-public/regulation-black-knight-1.jpg" width="100px"/> | <img src="img-public/wooden-white-bishop-1.jpg" width="70px"/> <img src="img-public/wooden-black-knight-1.jpg" width="100px"/>

----
### Resources

| Topic | Description | Link |
| --- | --- | --- |
| EDA | Analyze the annotated *Regulation Piece* dataset; join `Categories`, `Images`, and `Annotations` tables from COCO annotation. | [`eda_2.ipynb`](./eda_@.ipynb)|
| Cropping Script | use COCO annotations to crop individual pieces | [`crop_script_v1.py`](./crop_script_v1.py)|
| Training Notebooks | Fit different chess-piece datasets: <br> - Bing Image Search: 150 "X chess pieces", X = "bishop", "knight", "pawn" <br> - Regulation Pieces: from roboflow annotated dataset | <br> [Bing-Pieces-1](./clas_Exercise_1.ipynb) <br> [Regulation-Pieces-1](./clas_exercise_2.ipynb)|
| Production App | Run web-cam and make inference outside a notebook. <br> A simple client/server achitecture script. | [Link to repo `/app`](./app/)|
| Download Train Images | Use bing image api to download image datasets| [`bing_search_img.ipynb`](./bing_search_img.ipynb)|
| Scrap / Demo books | Various notebooks and scripts to demonstrate small points or explore concepts| |

---
### Dataset / EDA

Public dataset [here.](https://public.roboflow.ai/object-detection/chess-full) which includes images + annotations + data-augmentation


| Images / Annotations | Classes / Annotations |
| --- | --- |
| Roughly 300 standardized images of a chess board contain roughly 3,000 separate pieces .  | annotations with 100 - 200 annotations of each *{Piece-Class, Player-Color}*  |
|<img src="img-public/annotations_per_image.jpg" width="300px"/>|<img src="img-public/annotations_per_class.jpg" width="300px"/>|

------
### Build Cropped Images

Convert the photos of the full board with (possibly) many pieces to one bounded using the built using this script.

| Input | Output |
| --- | --- |
| <img src="img-public/full_board_1.jpg" width="400px"/> | <img src="img-public/crop_piece_1.jpg" width="150px"/> | 

### Train Models

Following the class exercise's direction, ResNet18 seems to work exceedingly well for this task: Almost perfect results for the Regulation pieces, and not results considering the Bing dataset pieces often are so different (e.g. from rare or exotic chess ets) as to not to be human recognizable either.

| Regulation Dataset | Bing Dataset 1 |
| --- | --- |
| <img src="img-public/cf_regulation_1.jpg" width="300px"/> | <img src="img-public/cf_bing_1.jpg" width="300px"/> | 



### Local real-time production app

This app runs "real time" meaning the webcame is always on and the app is making a prediction based on content (inside the yellow bounding box) every 2 seconds (configurable).

<img src="img-public/app1.gif"/>
<img src="img-public/cf_regulation_1.gif"/>

This uses a client/local-server architecture. The reason is I run my `fastai2` conda environment in WSL (linux virtual machine) but I can't use that for grabbing frames from my webcam. Therefore the client runs in windows and handles the `opencv` and webcam stuff and posts image data (via `requests`) to a flask server running in WSL which has a trained `learner` loaded and can predict the class of the piece.

Open Questions / Tasks:
 - how to display predict_proba for each class in matplotlib in a non-blocking fashion?
 - extend this to a remote server

