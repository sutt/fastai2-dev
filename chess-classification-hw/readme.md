# Chess Classification Homework

  *Dataset compliments of [Roboflow](https://roboflow.ai/)* 

Building an image classification pipeline with standard chess pieces...

*Questions / Investigations:*

 - Can you predict piece class on white peices when only trained on black?

 - How well does a model trained on standard peices perform on slighly non standard pieces?

 - What happens with piece occlusion?

----
### Dataset / EDA

Public dataset [here.](https://public.roboflow.ai/object-detection/chess-full) which includes images + annotations + data-augmentation

Roughly 300 standardized images of a chess board,  roughly 3,000 annotations with 100 - 200 annotations of each *{Piece-Class, Player-Color}*.

------
### First Stage: build cropped images

Convert the photos of the full board with [possibly] many pieces to one bounded using the built

