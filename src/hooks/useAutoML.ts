import { useState, useCallback } from 'react';
import { JobStatus, DataPreview, MLResults } from '../types/ml';
import { apiClient } from '../utils/api';

export const useAutoML = () => {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [results, setResults] = useState<MLResults | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const uploadFile = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.uploadFile(file);
      setJobId(response.job_id);
      setStatus({ status: 'uploaded', filename: file.name });
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getPreview = useCallback(async (jobId: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const previewData = await apiClient.getPreview(jobId);
      setPreview(previewData);
      return previewData;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Preview failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const startAnalysis = useCallback(async (jobId: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await apiClient.startAnalysis(jobId);
      setStatus({ status: 'analyzing' });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const pollStatus = useCallback(async (jobId: string) => {
    try {
      const statusData = await apiClient.getJobStatus(jobId);
      setStatus(statusData);
      
      if (statusData.status === 'completed') {
        const resultsData = await apiClient.getResults(jobId);
        setResults(resultsData);
      }
      
      return statusData;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Status check failed');
      throw err;
    }
  }, []);

  const downloadModel = useCallback(async (jobId: string) => {
    try {
      const blob = await apiClient.downloadModel(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `model_${jobId}.pkl`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model download failed');
      throw err;
    }
  }, []);

  const downloadReport = useCallback(async (jobId: string) => {
    try {
      const blob = await apiClient.downloadReport(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${jobId}.html`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report download failed');
      throw err;
    }
  }, []);

  const downloadCleanedData = useCallback(async (jobId: string) => {
    try {
      const blob = await apiClient.downloadCleanedData(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `cleaned_data_${jobId}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Cleaned data download failed');
      throw err;
    }
  }, []);

  const downloadMetrics = useCallback(async (jobId: string) => {
    try {
      const blob = await apiClient.downloadMetrics(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `metrics_${jobId}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Metrics download failed');
      throw err;
    }
  }, []);

  const downloadConfig = useCallback(async (jobId: string) => {
    try {
      const blob = await apiClient.downloadConfig(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pipeline_config_${jobId}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Configuration download failed');
      throw err;
    }
  }, []);

  const downloadAll = useCallback(async (jobId: string) => {
    try {
      const blob = await apiClient.downloadAll(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `automl_outputs_${jobId}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'ZIP download failed');
      throw err;
    }
  }, []);

  const reset = useCallback(() => {
    setJobId(null);
    setStatus(null);
    setPreview(null);
    setResults(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
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
  };
};