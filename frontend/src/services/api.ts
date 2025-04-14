import axios from 'axios';
import { Job, JobMatchResult, ApiResponse, ImprovementPlan } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const jobService = {
  matchJobs: async (file: File): Promise<ApiResponse<JobMatchResult>> => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await api.post<JobMatchResult>('/match-jobs', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return { data: response.data };
    } catch (error) {
      if (axios.isAxiosError(error)) {
        return { error: error.response?.data?.detail || 'An error occurred' };
      }
      return { error: 'An unexpected error occurred' };
    }
  },

  getImprovementPlan: async (evaluation: any): Promise<ApiResponse<ImprovementPlan[]>> => {
    console.log('Sending improvement plan request with evaluation:', evaluation);
    try {
      if (!evaluation?.gaps || !Array.isArray(evaluation.gaps) || evaluation.gaps.length === 0) {
        return { error: 'No gaps found in evaluation data' };
      }

      const response = await api.post<{ improvement_plan: ImprovementPlan[] }>('/improvement-plan', {
        gaps: evaluation.gaps,
      });
      console.log('Received improvement plan response:', response.data);
      return { data: response.data.improvement_plan };
    } catch (error) {
      console.error('Error in getImprovementPlan:', error);
      if (axios.isAxiosError(error)) {
        const errorMessage = error.response?.data?.detail || 
                           (Array.isArray(error.response?.data) ? error.response?.data[0]?.msg : 'An error occurred');
        return { error: errorMessage };
      }
      return { error: 'An unexpected error occurred' };
    }
  },

  searchJobs: async (query: string, limit: number = 10): Promise<ApiResponse<Job[]>> => {
    try {
      const response = await api.get<{ jobs: Job[] }>('/jobs', {
        params: { query, limit },
      });
      return { data: response.data.jobs };
    } catch (error) {
      if (axios.isAxiosError(error)) {
        return { error: error.response?.data?.detail || 'An error occurred' };
      }
      return { error: 'An unexpected error occurred' };
    }
  },

  getJobById: async (id: number): Promise<ApiResponse<Job>> => {
    try {
      const response = await api.get<Job>(`/jobs/${id}`);
      return { data: response.data };
    } catch (error) {
      if (axios.isAxiosError(error)) {
        return { error: error.response?.data?.detail || 'An error occurred' };
      }
      return { error: 'An unexpected error occurred' };
    }
  },
}; 