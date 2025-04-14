export interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  salary: string;
  description: string;
  requirements: string[];
  postedDate: string;
  similarityScore: number;
}

export interface Evaluation {
  score: number;
  explanation: string;
  strengths: string[];
  gaps: string[];
}

export interface JobMatchResult {
  top_matching_job: Job;
  similarity_score: number;
  evaluation: Evaluation;
}

export interface ImprovementPlan {
  gap: string;
  recommendations: {
    type: string;
    title: string;
    description: string;
    resources: string[];
    timeline: string;
    priority: 'high' | 'medium' | 'low';
  }[];
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
} 