import React from 'react';
import { Eye, Info, AlertTriangle } from 'lucide-react';
import { DataPreview as DataPreviewType } from '../types/ml';

interface DataPreviewProps {
  preview: DataPreviewType;
  onAnalyze: () => void;
  isLoading?: boolean;
}

export const DataPreview: React.FC<DataPreviewProps> = ({
  preview,
  onAnalyze,
  isLoading = false,
}) => {
  const missingValueCount = Object.values(preview.info.missing_values).reduce((sum, count) => sum + count, 0);
  const totalCells = preview.shape[0] * preview.shape[1];
  const missingPercentage = ((missingValueCount / totalCells) * 100).toFixed(1);

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Dataset Overview */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Info className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Dataset Overview</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{preview.shape[0]}</div>
            <div className="text-sm text-gray-600">Rows</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{preview.shape[1]}</div>
            <div className="text-sm text-gray-600">Columns</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{missingValueCount}</div>
            <div className="text-sm text-gray-600">Missing Values</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{missingPercentage}%</div>
            <div className="text-sm text-gray-600">Missing Rate</div>
          </div>
        </div>
        
        {missingValueCount > 0 && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <span className="text-sm font-medium text-yellow-800">
                Missing values detected - they will be handled automatically during preprocessing
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Column Information */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Column Information</h3>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Column
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Data Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Missing Values
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Missing %
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {preview.info.columns.map((column, index) => {
                const missing = preview.info.missing_values[column] || 0;
                const missingPct = ((missing / preview.shape[0]) * 100).toFixed(1);
                
                return (
                  <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {column}
                      {index === preview.info.columns.length - 1 && (
                        <span className="ml-2 inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          Target
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        preview.info.dtypes[column]?.includes('object') || preview.info.dtypes[column]?.includes('category')
                          ? 'bg-green-100 text-green-800'
                          : 'bg-blue-100 text-blue-800'
                      }`}>
                        {preview.info.dtypes[column]?.includes('object') || preview.info.dtypes[column]?.includes('category') ? 'Categorical' : 'Numerical'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {missing}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {missingPct}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Data Preview */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Eye className="h-5 w-5 text-green-600" />
          <h3 className="text-lg font-semibold text-gray-900">Data Preview</h3>
        </div>
        
        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-medium text-gray-900 mb-2">First 5 rows</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    {preview.info.columns.map((column, index) => (
                      <th key={index} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {preview.head.map((row, index) => (
                    <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      {preview.info.columns.map((column, colIndex) => (
                        <td key={colIndex} className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">
                          {row[column] !== null && row[column] !== undefined ? String(row[column]) : 'N/A'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="flex justify-center">
        <button
          onClick={onAnalyze}
          disabled={isLoading}
          className={`
            px-8 py-3 rounded-lg font-medium text-white transition-colors
            ${isLoading 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700'
            }
          `}
        >
          {isLoading ? 'Starting Analysis...' : 'Start AutoML Analysis'}
        </button>
      </div>
    </div>
  );
};