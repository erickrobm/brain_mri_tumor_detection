# Brain MRI Tumor Detection
Deep Learning model for brain MRI Tumor Detection.

## If you are using your GPU, follow this steps:

### About the Virual Environment

* Download the latest version of [**Visual Studio**](https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170) with an architecture compatible with your PC 
* Download the [**Anaconda Distribution**](https://www.anaconda.com/products/distribution)
* Once its dowloaded, open the **Anaconda Prompt** and create the **Anaconda Virtual Environment** with `conda create -n <name> python==3.8.8`. Feel free to select a name for the environment.
* Open the environment with `conda activate <name>`
* Install **CudaToolKit** and **Cudnn** with `conda install cudatoolkit=11.0 cudnn=8.0 -c=conda-forge`
* Install **Tensorflow-Gpu** with `pip install --upgrade tensorflow-gpu==2.4.1`

### How to verify if Tensorflow-GPU is correctly installed?

* On the **Anaconda Prompt** type `python`
* You may be able to run python scripts now, we´ll type the following lines:
* `import tensorflow as tf`
* `tf.test.is_gpu_available()`
* Once it finishes, you may be able to read at the end if it´s `True` that you´re using your GPU.

## If you aren´t using a GPU, follow this steps:

### About the Virual Environment

* Download the [**Anaconda Distribution**](https://www.anaconda.com/products/distribution)
* Once its dowloaded, open the **Anaconda Prompt** and create the **Anaconda Virtual Environment** with `conda create -n <name> python==3.8.8`. Feel free to select a name for the environment.
* Open the environment with `conda activate <name>`
* Install **Tensorflow** with `pip install --upgrade tensorflow==2.4.1`

## Installing the libraries 
* Once you have completed the previous steps, and still using the **Anaconda Prompt**, go to the folder `/brain_mri_tumor_detection`.
* Install all the libraries in the **Virtual Environment** with `pip install -r requirements.txt`

## Predicting Images

* Go to the folder `/tumor_detection/deployment` and then, type in the **Anaconda Prompt**: `uvicorn service:app --reload`