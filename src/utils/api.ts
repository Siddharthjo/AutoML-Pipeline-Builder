const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Utility function to parse CSV content
function parseCSV(csvText: string): { headers: string[], rows: any[][] } {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  const rows = lines.slice(1).map(line => 
    line.split(',').map(cell => {
      const trimmed = cell.trim();
      // Try to parse as number
      const num = parseFloat(trimmed);
      return isNaN(num) ? trimmed : num;
    })
  );
  return { headers, rows };
}

// Utility function to detect data types
function detectDataTypes(headers: string[], rows: any[][]): Record<string, string> {
  const types: Record<string, string> = {};
  
  headers.forEach((header, index) => {
    const values = rows.map(row => row[index]).filter(v => v !== null && v !== undefined && v !== '');
    
    if (values.length === 0) {
      types[header] = 'object';
      return;
    }
    
    const numericValues = values.filter(v => typeof v === 'number');
    const stringValues = values.filter(v => typeof v === 'string');
    
    if (numericValues.length > stringValues.length) {
      // Check if all numbers are integers
      const allIntegers = numericValues.every(v => Number.isInteger(v));
      types[header] = allIntegers ? 'int64' : 'float64';
    } else {
      types[header] = 'object';
    }
  });
  
  return types;
}

// Utility function to detect problem type
function detectProblemType(targetColumn: string, targetValues: any[]): 'classification' | 'regression' {
  const uniqueValues = [...new Set(targetValues.filter(v => v !== null && v !== undefined && v !== ''))];
  
  // If target has string values, it's classification
  if (uniqueValues.some(v => typeof v === 'string')) {
    return 'classification';
  }
  
  // If target has few unique numeric values (< 10 and < 5% of total), likely classification
  if (uniqueValues.length < 10 && uniqueValues.length / targetValues.length < 0.05) {
    return 'classification';
  }
  
  return 'regression';
}

// Utility function to calculate missing values
function calculateMissingValues(headers: string[], rows: any[][]): Record<string, number> {
  const missing: Record<string, number> = {};
  
  headers.forEach((header, index) => {
    const missingCount = rows.filter(row => 
      row[index] === null || row[index] === undefined || row[index] === '' || row[index] === 'N/A'
    ).length;
    missing[header] = missingCount;
  });
  
  return missing;
}

// Utility function to generate realistic metrics based on problem type and data quality
function generateRealisticMetrics(problemType: 'classification' | 'regression', dataQuality: number): Record<string, number> {
  // Data quality affects performance (0-1 scale)
  const basePerformance = 0.6 + (dataQuality * 0.35); // 60-95% range
  
  if (problemType === 'classification') {
    const accuracy = basePerformance + (Math.random() * 0.1 - 0.05);
    return {
      accuracy: Math.max(0.5, Math.min(0.99, accuracy)),
      precision: Math.max(0.5, Math.min(0.99, accuracy + (Math.random() * 0.05 - 0.025))),
      recall: Math.max(0.5, Math.min(0.99, accuracy + (Math.random() * 0.05 - 0.025))),
      f1_score: Math.max(0.5, Math.min(0.99, accuracy + (Math.random() * 0.03 - 0.015))),
      auc: Math.max(0.6, Math.min(0.99, accuracy + 0.05 + (Math.random() * 0.05)))
    };
  } else {
    const r2 = basePerformance + (Math.random() * 0.1 - 0.05);
    const targetRange = 1000; // Assume some reasonable target range for error metrics
    return {
      r2: Math.max(0.1, Math.min(0.95, r2)),
      rmse: (1 - r2) * targetRange * (0.5 + Math.random() * 0.5),
      mae: (1 - r2) * targetRange * (0.3 + Math.random() * 0.4),
      mape: (1 - r2) * 50 * (0.5 + Math.random() * 0.5)
    };
  }
}

export class MLApiClient {
  private baseUrl: string;
  private datasetCache: Map<string, any> = new Map();

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async uploadFile(file: File): Promise<{ job_id: string; dataset_info: any }> {
    // Mock response for demo purposes in Bolt environment
    if (this.baseUrl.includes('localhost')) {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const csvText = e.target?.result as string;
            const { headers, rows } = parseCSV(csvText);
            const dataTypes = detectDataTypes(headers, rows);
            const missingValues = calculateMissingValues(headers, rows);
            
            const jobId = 'demo-' + Math.random().toString(36).substr(2, 9);
            
            // Cache the dataset for later use
            this.datasetCache.set(jobId, {
              headers,
              rows,
              dataTypes,
              missingValues,
              csvText
            });
            
            setTimeout(() => {
              resolve({
                job_id: jobId,
                dataset_info: {
                  shape: [rows.length, headers.length],
                  columns: headers,
                  missing_values: missingValues,
                  data_types: dataTypes
                }
              });
            }, 1000);
          } catch (error) {
            console.error('Error parsing CSV:', error);
            // Fallback to default data
            const jobId = 'demo-' + Math.random().toString(36).substr(2, 9);
            setTimeout(() => {
              resolve({
                job_id: jobId,
                dataset_info: {
                  shape: [100, 4],
                  columns: ['feature1', 'feature2', 'feature3', 'target'],
                  missing_values: { feature1: 0, feature2: 0, feature3: 0, target: 0 },
                  data_types: { feature1: 'float64', feature2: 'float64', feature3: 'float64', target: 'object' }
                }
              });
            }, 1000);
          }
        };
        reader.readAsText(file);
      });
    }

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getPreview(jobId: string, rows: number = 5): Promise<any> {
    // Mock response for demo purposes in Bolt environment
    if (this.baseUrl.includes('localhost')) {
      return new Promise((resolve) => {
        const cachedData = this.datasetCache.get(jobId);
        
        if (cachedData) {
          const { headers, rows: allRows, dataTypes, missingValues } = cachedData;
          
          // Convert rows back to objects
          const head = allRows.slice(0, rows).map((row: any[]) => {
            const obj: any = {};
            headers.forEach((header: string, index: number) => {
              obj[header] = row[index];
            });
            return obj;
          });
          
          const tail = allRows.slice(-rows).map((row: any[]) => {
            const obj: any = {};
            headers.forEach((header: string, index: number) => {
              obj[header] = row[index];
            });
            return obj;
          });
          
          setTimeout(() => {
            resolve({
              head,
              tail,
              shape: [allRows.length, headers.length],
              info: {
                columns: headers,
                dtypes: dataTypes,
                missing_values: missingValues
              }
            });
          }, 500);
        } else {
          // Fallback preview
          setTimeout(() => {
            resolve({
              head: [
                { feature1: 5.1, feature2: 3.5, feature3: 1.4, target: 'A' },
                { feature1: 4.9, feature2: 3.0, feature3: 1.4, target: 'A' },
                { feature1: 4.7, feature2: 3.2, feature3: 1.3, target: 'B' }
              ],
              tail: [
                { feature1: 6.7, feature2: 3.0, feature3: 5.2, target: 'C' },
                { feature1: 6.3, feature2: 2.5, feature3: 5.0, target: 'C' },
                { feature1: 6.5, feature2: 3.0, feature3: 5.2, target: 'C' }
              ],
              shape: [100, 4],
              info: {
                columns: ['feature1', 'feature2', 'feature3', 'target'],
                dtypes: { feature1: 'float64', feature2: 'float64', feature3: 'float64', target: 'object' },
                missing_values: { feature1: 0, feature2: 0, feature3: 0, target: 0 }
              }
            });
          }, 500);
        }
      });
    }

    const response = await fetch(`${this.baseUrl}/preview/${jobId}?rows=${rows}`);
    
    if (!response.ok) {
      throw new Error(`Preview failed: ${response.statusText}`);
    }

    return response.json();
  }

  async startAnalysis(jobId: string): Promise<{ message: string; job_id: string }> {
    // Mock response for demo purposes in Bolt environment
    if (this.baseUrl.includes('localhost')) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            message: 'Analysis started',
            job_id: jobId
          });
        }, 500);
      });
    }

    const response = await fetch(`${this.baseUrl}/analyze/${jobId}`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getJobStatus(jobId: string): Promise<any> {
    // Mock progressive status updates for demo purposes in Bolt environment
    if (this.baseUrl.includes('localhost')) {
      return new Promise((resolve) => {
        const now = Date.now();
        const startTimeKey = `analysisStartTime_${jobId}`;
        const elapsed = now - ((window as any)[startTimeKey] || 0);
        
        if (!((window as any)[startTimeKey])) {
          (window as any)[startTimeKey] = now;
        }

        const cachedData = this.datasetCache.get(jobId);
        let problemType = 'classification';
        
        if (cachedData) {
          const { headers, rows } = cachedData;
          const targetColumn = headers[headers.length - 1];
          const targetValues = rows.map((row: any[]) => row[headers.length - 1]);
          problemType = detectProblemType(targetColumn, targetValues);
        }

        setTimeout(() => {
          if (elapsed < 2000) {
            resolve({
              status: 'analyzing',
              progress: 10,
              current_step: 'Data Cleaning',
              problem_type: problemType
            });
          } else if (elapsed < 4000) {
            resolve({
              status: 'running',
              progress: 30,
              current_step: 'Model Training',
              problem_type: problemType
            });
          } else if (elapsed < 6000) {
            resolve({
              status: 'running',
              progress: 70,
              current_step: 'Model Evaluation',
              problem_type: problemType
            });
          } else {
            resolve({
              status: 'completed',
              progress: 100,
              current_step: 'Generating Report',
              problem_type: problemType
            });
          }
        }, 500);
      });
    }

    const response = await fetch(`${this.baseUrl}/status/${jobId}`);
    
    if (!response.ok) {
      throw new Error(`Status check failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getResults(jobId: string): Promise<any> {
    // Mock results for demo purposes in Bolt environment
    if (this.baseUrl.includes('localhost')) {
      return new Promise((resolve) => {
        const cachedData = this.datasetCache.get(jobId);
        let problemType = 'classification';
        let metrics = {};
        
        if (cachedData) {
          const { headers, rows, missingValues } = cachedData;
          const targetColumn = headers[headers.length - 1];
          const targetValues = rows.map((row: any[]) => row[headers.length - 1]);
          problemType = detectProblemType(targetColumn, targetValues);
          
          // Calculate data quality based on missing values and dataset size
          const totalCells = headers.length * rows.length;
          const totalMissing = Object.values(missingValues).reduce((sum: number, count: number) => sum + count, 0);
          const dataQuality = 1 - (totalMissing / totalCells);
          
          metrics = generateRealisticMetrics(problemType, dataQuality);
        } else {
          // Fallback metrics
          metrics = {
            accuracy: 0.85,
            precision: 0.84,
            recall: 0.86,
            f1_score: 0.85
          };
        }
        
        setTimeout(() => {
          resolve({
            model_path: `/models/model_${jobId}.pkl`,
            metrics,
            visualizations: {
              confusion_matrix: `/reports/confusion_matrix_${jobId}.png`,
              roc_curve: `/reports/roc_curve_${jobId}.png`,
              prediction_distribution: `/reports/prediction_distribution_${jobId}.png`,
              ...(problemType === 'regression' && {
                actual_vs_predicted: `/reports/actual_vs_predicted_${jobId}.png`,
                residuals: `/reports/residuals_${jobId}.png`
              })
            },
            report_path: `/reports/report_${jobId}.html`,
            problem_type: problemType,
            completed_at: new Date().toISOString()
          });
        }, 500);
      });
    }

    const response = await fetch(`${this.baseUrl}/results/${jobId}`);
    
    if (!response.ok) {
      throw new Error(`Results fetch failed: ${response.statusText}`);
    }

    return response.json();
  }

  async downloadModel(jobId: string): Promise<Blob> {
    // Mock download for demo purposes in Bolt environment
    if (this.baseUrl.includes('localhost')) {
      return new Promise((resolve) => {
        setTimeout(() => {
          const cachedData = this.datasetCache.get(jobId);
          const modelInfo = cachedData ? 
            `# AutoML Model for dataset with ${cachedData.headers.length} features\n# Problem type: ${cachedData.problemType || 'classification'}\n# Generated: ${new Date().toISOString()}` :
            '# Mock AutoML model file';
          
          const mockData = new Blob([modelInfo], { type: 'application/octet-stream' });
          resolve(mockData);
        }, 500);
      });
    }

    const response = await fetch(`${this.baseUrl}/download/model/${jobId}`);
    
    if (!response.ok) {
      throw new Error(`Model download failed: ${response.statusText}`);
    }

    return response.blob();
  }

  async downloadReport(jobId: string): Promise<Blob> {
    // Mock download for demo purposes in Bolt environment
    if (this.baseUrl.includes('localhost')) {
      return new Promise((resolve) => {
        const cachedData = this.datasetCache.get(jobId);
        
        let reportContent = `
          <!DOCTYPE html>
          <html>
          <head>
            <title>AutoML Analysis Report</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              .header { background: #f4f4f4; padding: 20px; border-radius: 8px; }
              .metric { display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 4px; }
              table { border-collapse: collapse; width: 100%; margin: 20px 0; }
              th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
              th { background-color: #f2f2f2; }
            </style>
          </head>
          <body>
            <div class="header">
              <h1>AutoML Analysis Report</h1>
              <p>Generated on: ${new Date().toLocaleString()}</p>
            </div>
        `;
        
        if (cachedData) {
          const { headers, rows, dataTypes, missingValues } = cachedData;
          const targetColumn = headers[headers.length - 1];
          const targetValues = rows.map((row: any[]) => row[headers.length - 1]);
          const problemType = detectProblemType(targetColumn, targetValues);
          
          reportContent += `
            <h2>Dataset Information</h2>
            <p><strong>Shape:</strong> ${rows.length} rows × ${headers.length} columns</p>
            <p><strong>Problem Type:</strong> ${problemType}</p>
            <p><strong>Target Column:</strong> ${targetColumn}</p>
            
            <h2>Column Information</h2>
            <table>
              <tr><th>Column</th><th>Data Type</th><th>Missing Values</th></tr>
              ${headers.map(col => `
                <tr>
                  <td>${col}</td>
                  <td>${dataTypes[col]}</td>
                  <td>${missingValues[col] || 0}</td>
                </tr>
              `).join('')}
            </table>
            
            <h2>Sample Data</h2>
            <table>
              <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
              ${rows.slice(0, 5).map((row: any[]) => `
                <tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>
              `).join('')}
            </table>
          `;
        }
        
        reportContent += `
            <p><em>This is a demo report generated in the Bolt environment.</em></p>
          </body>
          </html>
        `;
        
        setTimeout(() => {
          const mockData = new Blob([reportContent], { type: 'text/html' });
          resolve(mockData);
        }, 500);
      });
    }

    const response = await fetch(`${this.baseUrl}/download/report/${jobId}`);
    
    if (!response.ok) {
      throw new Error(`Report download failed: ${response.statusText}`);
    }

    return response.blob();
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    // Mock health check for demo purposes in Bolt environment
    if (this.baseUrl.includes('localhost')) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            status: 'healthy',
            timestamp: new Date().toISOString()
          });
        }, 200);
      });
    }

    const response = await fetch(`${this.baseUrl}/health`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }
}

export const apiClient = new MLApiClient();