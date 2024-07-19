stabilityai/sdxl-turbo
https://huggingface.co/stabilityai/sdxl-turbo

Limitations

The generated images are of a fixed resolution (512x512 pix), and the model does not achieve perfect photorealism.

The model cannot render legible text.

Faces and people in general may not be generated properly.

The autoencoding part of the model is lossy.


runwayml/stable-diffusion-v1-5
https://huggingface.co/runwayml/stable-diffusion-v1-5

Limitations

The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”

The model was trained mainly with English captions and will not work as well in other languages.

The model was trained on a large-scale dataset LAION-5B which contains adult material and is not fit for product use without additional safety mechanisms and considerations.

No additional measures were used to deduplicate the dataset. As a result, we observe some degree of memorization for images that are duplicated in the training data. The training data can be searched at https://rom1504.github.io/clip-retrieval/ to possibly assist in the detection of memorized images.

prompt:

girl with brown dog hair, thick glasses, smiling

Portrait of The Joker halloween costume, face painting, with , glare pose, detailed