export interface DatasetInfo {
  shape: [number, number];
  columns: string[];
  missing_values: Record<string, number>;
  data_types: Record<string, string>;
}

export interface JobStatus {
  status: 'uploaded' | 'analyzing' | 'running' | 'completed' | 'failed';
  progress?: number;
  current_step?: string;
  error?: string;
  filename?: string;
  results?: MLResults;
}

export interface MLResults {
  model_path: string;
  metrics: Record<string, number>;
  visualizations: Record<string, string>;
  report_path: string;
  problem_type: 'classification' | 'regression';
  completed_at: string;
}

export interface DataPreview {
  head: Record<string, any>[];
  tail: Record<string, any>[];
  shape: [number, number];
  info: {
    columns: string[];
    dtypes: Record<string, string>;
    missing_values: Record<string, number>;
  };
}

export interface FeatureImportance {
  [feature: string]: number;
}