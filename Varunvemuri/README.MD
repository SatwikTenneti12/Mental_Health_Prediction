# MIND-MAP

# Dataset Upload Instructions for 3T fMRI Dataset

## Dataset Source:
We are using the **3T fMRI Dataset** from Kaggle. You can find the dataset here: [3T fMRI Dataset](https://www.kaggle.com/datasets/mathurinache/3t-fmri-dataset).

## Step-by-Step Guide to Download the Dataset Using Kaggle API

### Step 1: Set Up the Kaggle API

To download the dataset from Kaggle, you'll first need to set up the Kaggle API.

1. **Get the Kaggle API Key**:
   - Log in to [Kaggle](https://www.kaggle.com/).
   - Navigate to **My Account**.
   - Scroll down to the **API** section and click **Create New API Token**. This will download a file called `kaggle.json`.

2. **Upload `kaggle.json` to Google Colab**:
   Upload the `kaggle.json` file to your Colab or Jupyter environment using the following code:
   ```python
   from google.colab import files
   files.upload()  # This will prompt you to upload the kaggle.json file

