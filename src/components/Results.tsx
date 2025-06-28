import React from 'react';
import { 
  Download, 
  Trophy, 
  TrendingUp, 
  BarChart3, 
  FileText,
  Target,
  Zap
} from 'lucide-react';
import { MLResults } from '../types/ml';

interface ResultsProps {
  results: MLResults;
  onDownloadModel: () => void;
  onDownloadReport: () => void;
}

export const Results: React.FC<ResultsProps> = ({
  results,
  onDownloadModel,
  onDownloadReport,
}) => {
  const formatMetric = (key: string, value: number) => {
    if (key.includes('score') || key.includes('accuracy') || key.includes('precision') || key.includes('recall') || key.includes('f1')) {
      return `${(value * 100).toFixed(2)}%`;
    }
    return value.toFixed(4);
  };

  const getMetricIcon = (key: string) => {
    if (key.includes('accuracy') || key.includes('score')) return <Target className="h-4 w-4" />;
    if (key.includes('precision') || key.includes('recall')) return <TrendingUp className="h-4 w-4" />;
    return <BarChart3 className="h-4 w-4" />;
  };

  const getMetricColor = (key: string, value: number) => {
    if (key.includes('accuracy') || key.includes('f1') || key.includes('r2')) {
      if (value > 0.9) return 'text-green-600 bg-green-50';
      if (value > 0.7) return 'text-blue-600 bg-blue-50';
      return 'text-orange-600 bg-orange-50';
    }
    return 'text-gray-600 bg-gray-50';
  };

  const primaryMetrics = Object.entries(results.metrics).slice(0, 4);
  const secondaryMetrics = Object.entries(results.metrics).slice(4);

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Success Header */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 border border-green-200">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            <Trophy className="h-8 w-8 text-green-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Model Training Complete!</h2>
            <p className="text-gray-600 mt-1">
              Your AutoML pipeline has successfully trained and evaluated a {results.problem_type} model.
            </p>
          </div>
        </div>
      </div>

      {/* Primary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {primaryMetrics.map(([key, value]) => (
          <div key={key} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center space-x-3">
              <div className={`p-2 rounded-lg ${getMetricColor(key, value)}`}>
                {getMetricIcon(key)}
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {formatMetric(key, value)}
                </div>
                <div className="text-sm text-gray-600 capitalize">
                  {key.replace(/_/g, ' ')}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Secondary Metrics */}
      {secondaryMetrics.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Additional Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {secondaryMetrics.map(([key, value]) => (
              <div key={key} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-700 capitalize">
                  {key.replace(/_/g, ' ')}
                </span>
                <span className="text-sm font-semibold text-gray-900">
                  {formatMetric(key, value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Performance Interpretation */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Zap className="h-5 w-5 text-purple-600" />
          <h3 className="text-lg font-semibold text-gray-900">Performance Summary</h3>
        </div>
        
        <div className="prose text-gray-700">
          {results.problem_type === 'classification' ? (
            <div className="space-y-3">
              <p>
                <strong>Accuracy:</strong> {formatMetric('accuracy', results.metrics.accuracy || 0)} of predictions were correct.
              </p>
              {results.metrics.f1_score && (
                <p>
                  <strong>F1-Score:</strong> {formatMetric('f1_score', results.metrics.f1_score)} represents the harmonic mean of precision and recall.
                </p>
              )}
              {results.metrics.auc && (
                <p>
                  <strong>AUC:</strong> {formatMetric('auc', results.metrics.auc)} shows the model's ability to distinguish between classes.
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <p>
                <strong>R² Score:</strong> {formatMetric('r2', results.metrics.r2 || 0)} of the variance in the target variable is explained by the model.
              </p>
              {results.metrics.rmse && (
                <p>
                  <strong>RMSE:</strong> {results.metrics.rmse.toFixed(4)} is the root mean squared error between predictions and actual values.
                </p>
              )}
              {results.metrics.mae && (
                <p>
                  <strong>MAE:</strong> {results.metrics.mae.toFixed(4)} is the mean absolute error between predictions and actual values.
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Visualizations */}
      {results.visualizations && Object.keys(results.visualizations).length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Visualizations</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(results.visualizations).map(([key, path]) => (
              <div key={key} className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2 capitalize">
                  {key.replace(/_/g, ' ')}
                </h4>
                <div className="bg-white rounded border border-gray-200 p-2">
                  <div className="text-sm text-gray-600 text-center py-8">
                    Visualization available in downloaded report
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Download Actions */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Download Results</h3>
        <div className="flex flex-col sm:flex-row gap-4">
          <button
            onClick={onDownloadModel}
            className="flex items-center justify-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Download className="h-4 w-4" />
            <span>Download Model (.pkl)</span>
          </button>
          
          <button
            onClick={onDownloadReport}
            className="flex items-center justify-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <FileText className="h-4 w-4" />
            <span>Download Report (.html)</span>
          </button>
        </div>
        
        <div className="mt-4 text-sm text-gray-600">
          <p>
            <strong>Model file:</strong> Use this .pkl file to make predictions on new data using Python and scikit-learn.
          </p>
          <p className="mt-1">
            <strong>Report file:</strong> Contains detailed analysis, visualizations, and model performance metrics.
          </p>
        </div>
      </div>

      {/* Completion Info */}
      <div className="bg-gray-50 rounded-lg p-4 text-center">
        <p className="text-sm text-gray-600">
          Analysis completed on {new Date(results.completed_at).toLocaleString()}
        </p>
      </div>
    </div>
  );
};