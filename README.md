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

