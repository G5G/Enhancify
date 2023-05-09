# Enhancify Video Super-Resolution Software

Enhancify is a video super-resolution software that uses the Lern algorithm to upscale video quality. This software can be used for enhancing video resolution

## Installation

To install Enhancify, follow these steps:

1. Download the latest version of Python from the official website.
2. Install the [Visual C++ Redistributable for Visual Studio 2019](https://aka.ms/vs/16/release/vc_redist.x64.exe).
3. Install the required dependencies using `pip install -r requirementsPIP.txt`.
4. If you want to use the GPU version, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and install the CUDA version of PyTorch either through Conda or pip.

## Usage

To use Enhancify, follow these steps:

1. Open the terminal or an IDE and run the command `python main.py`.
2. Click on "Test the Lern network" to access the upscaling tool.
3. Select the location of the video that you want to upscale, the output folder where the upscaled video will be saved, and the location of the model (the Enhancify folder should contain three model folders).
4. Choose the model that you want to use and click "Run" to start the upscaling process.

## Custom Training

If you want to train the Lern network yourself, follow these steps:

1. Download the REDS dataset from [here](https://seungjunnah.github.io/Datasets/reds.html).
2. Select the following folders: train_sharp, val_sharp, train_sharp_bicubic, and val_sharp_bicubic.
3. Specify the Epoch, learning rate (suggested 0.0001), the scale (4 if using the 4x REDS dataset), the batch size (100 is recommended, but experiment with larger batch sizes), and tick the shuffle box if you want the dataset to be randomized.
4. Click "Train" to start the training process.
5. The graph will update during training, with the green line representing validation and the red line representing training. The performance is measured in PSNR, so the higher the value, the better the performance.

## Contact

If you have any questions or issues with Enhancify, please contact us at [email address/website link/social media handle].
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/G5G/Enhancify">Enhancify</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/G5G">Vilius Radavicius</a> is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution-NonCommercial-ShareAlike 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>
