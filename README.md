# AutoML Pipeline Builder

A comprehensive web application that automatically builds machine learning pipelines from CSV datasets. Upload your data, let AI handle the preprocessing, model selection, and training, then download your trained model and analysis report.

## Features

- **Automatic Data Analysis**: Upload CSV files and get instant data quality insights
- **Intelligent Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **Problem Type Detection**: Automatically determines if your problem is classification or regression
- **AutoML Integration**: Uses Auto-sklearn or TPOT for automated model selection and hyperparameter tuning
- **Comprehensive Evaluation**: Generates performance metrics and visualizations
- **Easy Export**: Download trained models (.pkl) and detailed HTML reports

## Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Auto-sklearn/TPOT**: AutoML libraries for automated model selection
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib/Seaborn**: Data visualization
- **ydata-profiling**: Automated EDA report generation

### Frontend
- **React**: Modern JavaScript library for building user interfaces
- **TypeScript**: Type-safe JavaScript development
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Beautiful, customizable icons

## Installation

### Prerequisites
- Node.js (v16 or higher)
- Python (v3.8 or higher)
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd automl-pipeline-builder
   ```

2. **Set up Python virtual environment**
   ```bash
   cd backend
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the FastAPI server**
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install Node.js dependencies**
   ```bash
   npm install
   ```

2. **Start the development server**
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:5173`

## Usage

### Quick Start

1. **Upload Dataset**: Click "Choose CSV File" or drag & drop your CSV file
2. **Preview Data**: Review your dataset structure and column information
3. **Start Analysis**: Click "Start AutoML Analysis" to begin the automated pipeline
4. **Monitor Progress**: Watch the real-time progress of data cleaning, model training, and evaluation
5. **Download Results**: Get your trained model (.pkl) and analysis report (.html)

### Dataset Requirements

- **Format**: CSV files only
- **Structure**: First row should contain column headers
- **Size**: At least 2 columns (features + target)
- **Target**: The last column is automatically detected as the target variable

### Sample Datasets

Two sample datasets are included in the `sample_data/` directory:

- `classification_sample.csv`: Iris flower classification dataset
- `regression_sample.csv`: House price prediction dataset

## API Documentation

### Endpoints

- `POST /upload`: Upload CSV file and validate dataset
- `GET /preview/{job_id}`: Get dataset preview and statistics
- `POST /analyze/{job_id}`: Start AutoML analysis pipeline
- `GET /status/{job_id}`: Check analysis progress
- `GET /results/{job_id}`: Get final results and metrics
- `GET /download/model/{job_id}`: Download trained model
- `GET /download/report/{job_id}`: Download analysis report

### WebSocket Support

Real-time updates are provided through HTTP polling. The frontend automatically polls the status endpoint every 2 seconds during analysis.

## Model Types

### Classification
- **Algorithms**: Random Forest, Logistic Regression, SVM, and more via AutoML
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Visualizations**: Confusion Matrix, ROC Curve, Prediction Distribution

### Regression
- **Algorithms**: Random Forest, Linear Regression, SVR, and more via AutoML
- **Metrics**: R² Score, RMSE, MAE, MAPE
- **Visualizations**: Actual vs Predicted, Residual Plots, Distribution Comparison

## Data Preprocessing

The pipeline automatically handles:

- **Missing Values**: Imputation using median (numerical) or mode (categorical)
- **Categorical Encoding**: One-hot encoding for low cardinality, label encoding for high cardinality
- **Feature Scaling**: StandardScaler for numerical features
- **Data Validation**: Type checking, empty dataset detection, minimum column requirements

## AutoML Integration

### Auto-sklearn (Preferred)
- Automated algorithm selection and hyperparameter optimization
- Ensemble methods for improved performance
- Built-in cross-validation and model evaluation

### TPOT (Fallback)
- Genetic programming approach to AutoML
- Pipeline optimization including feature selection
- Scikit-learn compatible models

### Traditional ML (Fallback)
- Grid search over multiple algorithms
- Random Forest, Logistic Regression, SVM
- Basic hyperparameter tuning

## Deployment

### Local Development
```bash
# Terminal 1 - Backend
cd backend
python main.py

# Terminal 2 - Frontend
npm run dev
```

### Production Deployment

#### Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build
```

#### Render/Railway
1. Deploy backend as a Python web service
2. Deploy frontend as a static site
3. Configure CORS settings for production URLs

#### Hugging Face Spaces
1. Create a new Space with Gradio/Streamlit SDK
2. Upload backend code
3. Configure requirements.txt
4. Deploy automatically

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS Settings
CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]

# AutoML Settings
AUTOML_TIME_LIMIT=300  # seconds
AUTOML_MEMORY_LIMIT=3072  # MB

# File Upload Settings
MAX_FILE_SIZE=100  # MB
UPLOAD_DIRECTORY=uploads/
```

### Model Configuration

Modify AutoML settings in `backend/utils/model_training.py`:

```python
# Auto-sklearn settings
time_left_for_this_task=300,  # 5 minutes
per_run_time_limit=30,        # 30 seconds per model
memory_limit=3072,            # 3GB RAM

# TPOT settings
generations=5,                # Number of iterations
population_size=20,           # Models per generation
max_time_mins=5,              # Maximum time
```

## Troubleshooting

### Common Issues

1. **AutoML Library Installation**
   ```bash
   # If auto-sklearn fails to install
   pip install auto-sklearn --no-deps
   pip install scikit-learn==1.3.2
   ```

2. **Memory Issues**
   - Reduce dataset size for large files
   - Increase system memory allocation
   - Use traditional ML fallback instead

3. **CORS Errors**
   - Check frontend/backend URLs match
   - Verify CORS_ORIGINS configuration
   - Ensure both servers are running

4. **File Upload Issues**
   - Verify CSV format and encoding
   - Check file size limits
   - Ensure proper column headers

### Performance Optimization

- **Dataset Size**: Optimal size is 1K-100K rows
- **Feature Count**: Works best with 5-50 features
- **Time Limits**: Adjust AutoML time limits based on dataset size
- **Memory Usage**: Monitor system resources during training

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions:

- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting section

## Roadmap

- [ ] Support for more file formats (Excel, JSON, Parquet)
- [ ] Advanced feature engineering options
- [ ] Model interpretability features (SHAP values)
- [ ] Multi-target prediction support
- [ ] Real-time prediction API
- [ ] Model monitoring and drift detection
- [ ] Integration with cloud ML platforms

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [Auto-sklearn](https://automl.github.io/auto-sklearn/)
- UI components from [Tailwind CSS](https://tailwindcss.com/)
- Icons by [Lucide](https://lucide.dev/)