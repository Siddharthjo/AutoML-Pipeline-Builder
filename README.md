#  AutoML Pipeline Builder

A full-stack web application that automates the entire machine learning pipeline. Upload your dataset and receive a trained model, performance metrics, and a complete analysis report — without writing a single line of code.

---

##  Features

- 📂 Upload CSV datasets with instant validation
- 🔍 Auto-detect data types and problem type (classification/regression)
- 🧹 Automated data preprocessing (missing value imputation, encoding, scaling)
- ⚙️ Smart model selection using AutoML (Auto-sklearn / TPOT)
- 📈 Performance metrics and visualizations (ROC, Confusion Matrix, etc.)
- 📝 Download trained models and HTML reports
- 🔄 Real-time progress updates & intuitive UI

---

##  Architecture

### Frontend: `React + TypeScript`
- **React 18**: Component-based UI
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first responsive styling
- **Vite**: Fast build tool
- **Lucide React**: Consistent icons

### Backend: `Python + FastAPI`
- **FastAPI**: High-performance REST API
- **Uvicorn**: ASGI server
- **Pandas, NumPy**: Data handling
- **Scikit-learn**: ML preprocessing & base models
- **Auto-sklearn / TPOT**: Automated model optimization
- **Matplotlib, Seaborn**: Visualization
- **Jinja2**: HTML templating for reports

---

##  App Workflow

1. **Upload CSV**
2. **Auto Detect Columns & Data Types**
3. **Problem Type Detection**
4. **Data Cleaning (imputation, encoding, scaling)**
5. **Model Training via AutoML (Auto-sklearn → TPOT → fallback)**
6. **Evaluation + Visual Report**
7. **Download Model and Report**

---

##  Core Logic

### Data Type Detection
```python
def detect_data_types(headers, rows):
    # Infer numerical, categorical, or mixed types

Problem Type Detection

def detect_problem_type(target_values):
    if categorical: return 'classification'
    else: return 'regression'

Preprocessing Pipeline

class DataCleaner:
    # Imputation, encoding, scaling

AutoML Model Selection

if auto_sklearn_available:
    use_auto_sklearn()
elif tpot_available:
    use_tpot()
else:
    fallback_to_gridsearch()


⸻

 Mock vs Production

Feature	Mock Mode	Production Mode
Data Parsing	Client-side JS	FastAPI server-side
AutoML Training	Simulated (JS)	Real model training (Python)
Metrics Display	Random values	True metrics & plots
Model Download	Placeholder .pkl	Trained .pkl file


⸻

 Setup & Run

Prerequisites
	•	Node.js ≥ 18
	•	Python ≥ 3.9
	•	pip, virtualenv
	•	Optional: Docker

Clone Repo

git clone https://github.com/your-username/automl-pipeline-builder.git
cd automl-pipeline-builder

Run Backend

cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload

Run Frontend

cd frontend
npm install
npm run dev


⸻

 Environment Variables

Create a .env file in backend/ (optional for API keys or S3):

MODEL_STORAGE_PATH=models/
ENABLE_AUTO_SKLEARN=True
ENABLE_TPOT=True


⸻

 Future Improvements
	•	Plotly interactive charts
	•	Dataset profiling with ydata-profiling
	•	SHAP/LIME model explainability
	•	Multi-file dataset support
	•	Cloud deployment via Docker/Kubernetes

⸻


 Contributing

PRs are welcome! Please open an issue first to discuss changes or improvements.

⸻

 License

MIT License © 2025 Your Name

⸻

 Contact

For feedback, feature requests, or bugs, open an issue or reach out at youremail@example.com.

---

###  What to do next:

- Replace the screenshot links with your real image URLs (e.g., from Imgur or GitHub assets).
- Replace `your-username`, `youremail@example.com`, and repo links with your real GitHub info.
- Add this to your repository root as `README.md`.

Let me know if you want:
- A logo or banner image for the top
- Shields/badges (build, license, etc.)
- Docker deployment section  
- GitHub Pages or Vercel deployment steps for frontend

Want me to generate the `requirements.txt`, `docker-compose.yml`, or API schema too?
