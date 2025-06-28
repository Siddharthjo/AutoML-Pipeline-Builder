import React, { useCallback } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  isLoading?: boolean;
  error?: string | null;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileUpload,
  isLoading = false,
  error = null,
}) => {
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = Array.from(e.dataTransfer.files);
    const csvFile = files.find(file => file.name.endsWith('.csv'));
    
    if (csvFile) {
      onFileUpload(csvFile);
    }
  }, [onFileUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  }, [onFileUpload]);

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div
        className={`
          border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${isLoading ? 'border-blue-300 bg-blue-50' : 'border-gray-300 hover:border-blue-400'}
          ${error ? 'border-red-300 bg-red-50' : ''}
        `}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center space-y-4">
          {isLoading ? (
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          ) : (
            <Upload className="h-12 w-12 text-gray-400" />
          )}
          
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-gray-900">
              Upload your CSV dataset
            </h3>
            <p className="text-sm text-gray-600">
              Drag and drop your CSV file here, or click to browse
            </p>
          </div>
          
          <input
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            disabled={isLoading}
            className="hidden"
            id="file-upload"
          />
          
          <label
            htmlFor="file-upload"
            className={`
              inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white
              ${isLoading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
              }
              transition-colors
            `}
          >
            <FileText className="h-4 w-4 mr-2" />
            Choose CSV File
          </label>
          
          {error && (
            <div className="flex items-center space-x-2 text-red-600 text-sm mt-4">
              <AlertCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="mt-6 text-sm text-gray-600">
        <h4 className="font-medium mb-2">Requirements:</h4>
        <ul className="list-disc list-inside space-y-1">
          <li>File must be in CSV format</li>
          <li>First row should contain column headers</li>
          <li>Dataset should have at least 2 columns</li>
          <li>Target column will be auto-detected (typically the last column)</li>
        </ul>
      </div>
    </div>
  );
};