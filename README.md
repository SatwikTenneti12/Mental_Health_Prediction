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

```bash
git clone <your-repository-url>
cd project_mindmap

### Running the Application with Docker

1. **Build the Docker Image**:

   ```bash
   docker build -t gradio-survey-app .

