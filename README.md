# MIND-MAP

# Survey Analysis and Mental Health Prediction

This project focuses on analyzing survey responses and predicting mental health levels (Depression, Stress, and Anxiety) based on the collected data using pre-trained machine learning models. The application is built using Gradio for an interactive user interface and Docker for containerization.

## Project Structure

- **Dockerfile**: Used to build a Docker image for the project.
- **README.md**: Project documentation and setup instructions.
- **Train.ipynb**: Notebook for training the machine learning models.
- **codebook.txt**: Describes the dataset features.
- **data.csv**: The original dataset used for training the models.
- **survey_results.csv**: Stores user responses from the Gradio interface.
- **test.py**: Main application code for preprocessing, prediction, and integration with Gradio.

## Prerequisites

1. Docker installed on your system.
   - [Install Docker](https://docs.docker.com/get-docker/).
2. Access to the pre-trained models:
   - Download the models from [Google Drive](https://your-google-drive-link-here).
   - Place the models in the same directory as `test.py`.

## Setting Up the Project

### Clone the Repository

  
   git clone <your-repository-url>
   cd project_mindmap


### Running the Application with Docker

1. **Build the Docker Image**:

   ```bash
   docker build -t gradio-survey-app .

### Accessing the Dataset and Models

- **Dataset**:
  - Place `data.csv` in the project directory for training.
  - `survey_results.csv` will store the survey responses from users.
- **Models**:
  - Pre-trained models for Depression, Stress, and Anxiety (`random_forest_Depression.pkl`, `random_forest_stress.pkl`, `random_forest_anxiety.pkl`) must be placed in the project directory.
  - Download models from the provided Google Drive link.

---

## Training the Models

1. Open `Train.ipynb` in a Jupyter Notebook or any Python IDE.
2. Train the models using the provided `data.csv`.
3. Save the trained models in `.pkl` format.

---

## Application Functionality

- **Survey Questions**: Collects user responses for DASS, TIPI, and VCL questions.
- **Prediction**: Uses pre-trained machine learning models to predict:
  - Depression level
  - Stress level
  - Anxiety level
- **Validation**: Ensures all required responses are provided and checks for invalid inputs.

---


