# my-ai-cv

## Overview
`my-ai-cv` is a project implementing a Retrieve Augmented Generation pipeline using a quantized tiny LLM with GGUF, designed to work across any device. This enables the use of the model without directly pushing it to the repository, as it is available within the container.

An Hugging Face Space version 🤗 is available [here](https://huggingface.co/spaces/ilanaliouchouche/assistant).

## Getting Started

### Prerequisites
Before running the project, ensure you have Docker installed on your machine. If you do not have Docker, you can install it from [Docker's official site](https://www.docker.com/products/docker-desktop).


### Running the Project

There are two main ways to launch the chat interface:

#### Option 1: Using Docker
Run the following command to pull the image from the registry and start the service:

```bash
docker run -p 7860:7860 -d ghcr.io/ilanaliouchouche/my-cv
```

The model currently in the container is: [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)

#### Option 2: Cloning and Running Locally
1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/yourusername/my-ai-cv.git
   cd my-ai-cv/assistant
   pip install -r requirements.txt
   ```

2. Ensure the `assistant/models` directory is created and populated as the model files are not pushed to the repository due to their size. Place the model files inside this directory.

3. Modify the `.env` file to match the model configurations, specifically the `LLM_PATH` should point to the model file within the `assistant/models` directory.

4. Run the application:
    ```bash
    python app.py
    ```

## Configuration & Customization

| Section        | Description                                                                                                                                                        |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Model**      | The project uses the `microsoft/Phi-3-mini-4k-instruct-gguf` model, specified in the `.env` file.                                                                                |
| **LLM_PATH**   | It is crucial to adjust the `LLM_PATH` variable in the `.env` file to point to the correct model file in the `assistant/models` directory.                         |
| **Template**   | Inside `assistant/ChatbotModel`, it is strongly recommended to update the static `TEMPLATE` variable with instructions tailored to the model used.             |
| **Chroma**   | Replace the content of `assistant/chromadb` with your own vector database containing the information you want to share.             |

### Customization 
If you wish to adapt the project for your personal CV, please **FORK** the repo. You must ensure the `LLM_PATH` is correctly set in your `.env` file and update the `ChatbotModel` class as necessary to align with your model's specifications.

