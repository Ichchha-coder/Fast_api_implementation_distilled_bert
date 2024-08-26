
## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Place your model files in the `models/` directory:**

    Ensure that `config.json` and `model.safetensors` (or `pytorch_model.bin`) are located in the `models/` directory.

5. **Run the FastAPI server:**

    ```bash
    uvicorn app.main:app --reload
    ```

6. **Access the API:**

    - API documentation is available at `http://127.0.0.1:8000/docs`.
    - Use the `/predict` endpoint to submit text for sentiment analysis.

## Notes

- This project uses DistilBERT for sentiment analysis.
- Ensure that your model files are correctly placed in the `models/` directory.
