import React, { useState, useEffect } from 'react';
import { Brain, RefreshCw, AlertCircle } from 'lucide-react';
import { FileUpload } from './components/FileUpload';
import { DataPreview } from './components/DataPreview';
import { ModelTraining } from './components/ModelTraining';
import { Results } from './components/Results';
import { useAutoML } from './hooks/useAutoML';

function App() {
  const {
    jobId,
    status,
    preview,
    results,
    isLoading,
    error,
    uploadFile,
    getPreview,
    startAnalysis,
    pollStatus,
    downloadModel,
    downloadReport,
    downloadCleanedData,
    downloadMetrics,
    downloadConfig,
    downloadAll,
    reset,
  } = useAutoML();

  const [currentStep, setCurrentStep] = useState<'upload' | 'preview' | 'training' | 'results'>('upload');

  useEffect(() => {
    if (status?.status === 'uploaded' && jobId) {
      setCurrentStep('preview');
    } else if (status?.status === 'analyzing' || status?.status === 'running') {
      setCurrentStep('training');
    } else if (status?.status === 'completed') {
      setCurrentStep('results');
    }
  }, [status, jobId]);

  const handleFileUpload = async (file: File) => {
    try {
      const response = await uploadFile(file);
      if (response.job_id) {
        await getPreview(response.job_id);
      }
    } catch (err) {
      console.error('Upload failed:', err);
    }
  };

  const handleStartAnalysis = async () => {
    if (!jobId) return;
    
    try {
      await startAnalysis(jobId);
    } catch (err) {
      console.error('Analysis failed:', err);
    }
  };

  const handleStatusUpdate = async (jobId: string) => {
    try {
      await pollStatus(jobId);
    } catch (err) {
      console.error('Status update failed:', err);
    }
  };

  const handleDownloadModel = async () => {
    if (!jobId) return;
    try {
      await downloadModel(jobId);
    } catch (err) {
      console.error('Model download failed:', err);
    }
  };

  const handleDownloadReport = async () => {
    if (!jobId) return;
    try {
      await downloadReport(jobId);
    } catch (err) {
      console.error('Report download failed:', err);
    }
  };

  const handleDownloadCleanedData = async () => {
    if (!jobId) return;
    try {
      await downloadCleanedData(jobId);
    } catch (err) {
      console.error('Cleaned data download failed:', err);
    }
  };

  const handleDownloadMetrics = async () => {
    if (!jobId) return;
    try {
      await downloadMetrics(jobId);
    } catch (err) {
      console.error('Metrics download failed:', err);
    }
  };

  const handleDownloadConfig = async () => {
    if (!jobId) return;
    try {
      await downloadConfig(jobId);
    } catch (err) {
      console.error('Configuration download failed:', err);
    }
  };

  const handleDownloadAll = async () => {
    if (!jobId) return;
    try {
      await downloadAll(jobId);
    } catch (err) {
      console.error('ZIP download failed:', err);
    }
  };

  const handleReset = () => {
    reset();
    setCurrentStep('upload');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <Brain className="h-8 w-8 text-blue-600" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">AutoML Pipeline Builder</h1>
                <p className="text-sm text-gray-600">Upload, analyze, and train ML models automatically</p>
              </div>
            </div>
            
            {currentStep !== 'upload' && (
              <button
                onClick={handleReset}
                className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                <span>New Analysis</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Progress Indicator */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-center py-4">
            <div className="flex items-center space-x-4">
              {['upload', 'preview', 'training', 'results'].map((step, index) => {
                const isActive = currentStep === step;
                const isCompleted = ['upload', 'preview', 'training', 'results'].indexOf(currentStep) > index;
                
                return (
                  <React.Fragment key={step}>
                    <div className={`
                      flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium
                      ${isActive ? 'bg-blue-600 text-white' : 
                        isCompleted ? 'bg-green-600 text-white' : 
                        'bg-gray-200 text-gray-600'}
                    `}>
                      {index + 1}
                    </div>
                    {index < 3 && (
                      <div className={`w-12 h-1 ${isCompleted ? 'bg-green-600' : 'bg-gray-200'}`} />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-red-600" />
              <h3 className="font-medium text-red-900">Error</h3>
            </div>
            <p className="text-sm text-red-700 mt-2">{error}</p>
          </div>
        )}

        {/* Step Content */}
        {currentStep === 'upload' && (
          <FileUpload 
            onFileUpload={handleFileUpload}
            isLoading={isLoading}
            error={error}
          />
        )}

        {currentStep === 'preview' && preview && (
          <DataPreview
            preview={preview}
            onAnalyze={handleStartAnalysis}
            isLoading={isLoading}
          />
        )}

        {currentStep === 'training' && status && jobId && (
          <ModelTraining
            jobId={jobId}
            status={status}
            onStatusUpdate={handleStatusUpdate}
          />
        )}

        {currentStep === 'results' && results && (
          <Results
            results={results}
            onDownloadModel={handleDownloadModel}
            onDownloadReport={handleDownloadReport}
            onDownloadCleanedData={handleDownloadCleanedData}
            onDownloadMetrics={handleDownloadMetrics}
            onDownloadConfig={handleDownloadConfig}
            onDownloadAll={handleDownloadAll}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p>AutoML Pipeline Builder - Automated Machine Learning Made Simple</p>
            <p className="mt-1">Upload your data, let AI do the work, download comprehensive outputs.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;