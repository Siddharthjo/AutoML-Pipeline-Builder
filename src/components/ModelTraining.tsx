import React, { useEffect } from 'react';
import { Brain, Clock, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { JobStatus } from '../types/ml';

interface ModelTrainingProps {
  jobId: string;
  status: JobStatus;
  onStatusUpdate: (jobId: string) => void;
}

export const ModelTraining: React.FC<ModelTrainingProps> = ({
  jobId,
  status,
  onStatusUpdate,
}) => {
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (status.status === 'analyzing' || status.status === 'running') {
      interval = setInterval(() => {
        onStatusUpdate(jobId);
      }, 2000); // Poll every 2 seconds
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [jobId, status.status, onStatusUpdate]);

  const getStatusIcon = () => {
    switch (status.status) {
      case 'analyzing':
      case 'running':
        return <Loader2 className="h-6 w-6 text-blue-600 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-6 w-6 text-green-600" />;
      case 'failed':
        return <AlertCircle className="h-6 w-6 text-red-600" />;
      default:
        return <Clock className="h-6 w-6 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'analyzing':
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const steps = [
    { id: 'analyzing', name: 'Data Analysis', description: 'Analyzing dataset structure and quality' },
    { id: 'cleaning', name: 'Data Cleaning', description: 'Handling missing values and encoding features' },
    { id: 'detection', name: 'Problem Detection', description: 'Determining if this is classification or regression' },
    { id: 'training', name: 'Model Training', description: 'Training and tuning multiple models with AutoML' },
    { id: 'evaluation', name: 'Model Evaluation', description: 'Evaluating model performance and generating metrics' },
    { id: 'reporting', name: 'Report Generation', description: 'Creating visualizations and analysis report' },
  ];

  const getCurrentStepIndex = () => {
    if (status.current_step?.includes('Data Cleaning')) return 1;
    if (status.current_step?.includes('Problem Detection')) return 2;
    if (status.current_step?.includes('Model Training')) return 3;
    if (status.current_step?.includes('Model Evaluation')) return 4;
    if (status.current_step?.includes('Generating Report')) return 5;
    return 0;
  };

  const currentStepIndex = getCurrentStepIndex();

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Status Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-4">
          {getStatusIcon()}
          <div className="flex-1">
            <div className="flex items-center space-x-3">
              <h3 className="text-lg font-semibold text-gray-900">AutoML Pipeline</h3>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor()}`}>
                {status.status.charAt(0).toUpperCase() + status.status.slice(1)}
              </span>
            </div>
            {status.current_step && (
              <p className="text-sm text-gray-600 mt-1">{status.current_step}</p>
            )}
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-gray-900">
              {status.progress || 0}%
            </div>
            <div className="text-sm text-gray-600">Complete</div>
          </div>
        </div>
        
        {/* Progress Bar */}
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${status.progress || 0}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Steps */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Pipeline Steps</h4>
        
        <div className="space-y-4">
          {steps.map((step, index) => {
            const isActive = index === currentStepIndex && (status.status === 'analyzing' || status.status === 'running');
            const isCompleted = index < currentStepIndex || status.status === 'completed';
            const isFailed = status.status === 'failed' && index === currentStepIndex;
            
            return (
              <div key={step.id} className="flex items-start space-x-4">
                <div className={`
                  flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                  ${isCompleted ? 'bg-green-100 text-green-600' : 
                    isActive ? 'bg-blue-100 text-blue-600' : 
                    isFailed ? 'bg-red-100 text-red-600' : 
                    'bg-gray-100 text-gray-400'}
                `}>
                  {isCompleted ? (
                    <CheckCircle className="h-4 w-4" />
                  ) : isActive ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : isFailed ? (
                    <AlertCircle className="h-4 w-4" />
                  ) : (
                    <span className="text-sm font-medium">{index + 1}</span>
                  )}
                </div>
                
                <div className="flex-1">
                  <h5 className={`font-medium ${
                    isActive ? 'text-blue-900' : 
                    isCompleted ? 'text-green-900' : 
                    isFailed ? 'text-red-900' : 
                    'text-gray-900'
                  }`}>
                    {step.name}
                  </h5>
                  <p className={`text-sm ${
                    isActive ? 'text-blue-600' : 
                    isCompleted ? 'text-green-600' : 
                    isFailed ? 'text-red-600' : 
                    'text-gray-600'
                  }`}>
                    {step.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Problem Type Detection */}
      {status.problem_type && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-3">
            <Brain className="h-5 w-5 text-purple-600" />
            <h4 className="font-semibold text-gray-900">Problem Type Detected</h4>
          </div>
          <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
            status.problem_type === 'classification' 
              ? 'bg-green-100 text-green-800' 
              : 'bg-blue-100 text-blue-800'
          }`}>
            {status.problem_type.charAt(0).toUpperCase() + status.problem_type.slice(1)}
          </div>
          <p className="text-sm text-gray-600 mt-2">
            {status.problem_type === 'classification' 
              ? 'This is a classification problem. The model will predict discrete categories.'
              : 'This is a regression problem. The model will predict continuous numerical values.'
            }
          </p>
        </div>
      )}

      {/* Error Display */}
      {status.status === 'failed' && status.error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <h4 className="font-medium text-red-900">Training Failed</h4>
          </div>
          <p className="text-sm text-red-700 mt-2">{status.error}</p>
        </div>
      )}
    </div>
  );
};