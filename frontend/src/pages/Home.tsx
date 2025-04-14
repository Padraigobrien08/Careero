import React, { useState } from 'react';
import { Box, Container, Typography, CircularProgress, Alert } from '@mui/material';
import { FileUpload } from '../components/FileUpload';
import { JobCard } from '../components/JobCard';
import { ImprovementPlan } from '../components/ImprovementPlan';
import { jobService } from '../services/api';
import { JobMatchResult, ImprovementPlan as ImprovementPlanType } from '../types';

export const Home: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<JobMatchResult | null>(null);
  const [improvementPlan, setImprovementPlan] = useState<ImprovementPlanType[] | null>(null);
  const [loadingPlan, setLoadingPlan] = useState(false);

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setImprovementPlan(null);

    try {
      const response = await jobService.matchJobs(file);
      if (response.error) {
        setError(response.error);
      } else if (response.data) {
        console.log('Match jobs response:', response.data);
        setResult(response.data);
        // Get improvement plan after successful job match
        if (response.data.evaluation) {
          console.log('Evaluation data:', response.data.evaluation);
          console.log('Gaps in evaluation:', response.data.evaluation.gaps);
          setLoadingPlan(true);
          try {
            // Ensure we have gaps before making the request
            if (!response.data.evaluation.gaps || response.data.evaluation.gaps.length === 0) {
              setError('No gaps found in the evaluation. Please try again.');
              setLoadingPlan(false);
              return;
            }
            
            const planResponse = await jobService.getImprovementPlan(response.data.evaluation);
            if (planResponse.error) {
              console.error('Error getting improvement plan:', planResponse.error);
              setError(planResponse.error);
            } else if (planResponse.data) {
              console.log('Improvement plan received:', planResponse.data);
              setImprovementPlan(planResponse.data);
            }
          } catch (err) {
            console.error('Error getting improvement plan:', err);
            setError('Failed to generate improvement plan. Please try again.');
          } finally {
            setLoadingPlan(false);
          }
        }
      }
    } catch (err) {
      console.error('Error in handleFileUpload:', err);
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        Career Matcher
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" align="center" paragraph>
        Upload your resume to find matching jobs and get personalized career insights
      </Typography>

      <Box sx={{ my: 4 }}>
        <FileUpload onFileAccepted={handleFileUpload} />
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ my: 2 }}>
          {typeof error === 'string' ? error : 'An unexpected error occurred'}
        </Alert>
      )}

      {result && result.top_matching_job && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Best Match
          </Typography>
          <JobCard job={result.top_matching_job} showScore />

          {result.evaluation && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h5" gutterBottom>
                Evaluation
              </Typography>
              <Typography variant="body1" paragraph>
                Score: {result.evaluation.score}/100
              </Typography>
              <Typography variant="body1" paragraph>
                {result.evaluation.explanation}
              </Typography>

              {result.evaluation.strengths && result.evaluation.strengths.length > 0 && (
                <>
                  <Typography variant="h6" gutterBottom>
                    Strengths
                  </Typography>
                  <ul>
                    {result.evaluation.strengths.map((strength, index) => (
                      <li key={index}>{strength}</li>
                    ))}
                  </ul>
                </>
              )}

              {result.evaluation.gaps && result.evaluation.gaps.length > 0 && (
                <>
                  <Typography variant="h6" gutterBottom>
                    Areas for Improvement
                  </Typography>
                  <ul>
                    {result.evaluation.gaps.map((gap, index) => (
                      <li key={index}>{gap}</li>
                    ))}
                  </ul>
                </>
              )}
            </Box>
          )}
        </Box>
      )}

      {loadingPlan && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
          <Typography variant="body1" sx={{ ml: 2 }}>
            Generating improvement plan...
          </Typography>
        </Box>
      )}

      {improvementPlan && improvementPlan.length > 0 ? (
        <ImprovementPlan plan={improvementPlan} />
      ) : result?.evaluation?.gaps && result.evaluation.gaps.length > 0 && !loadingPlan && (
        <Alert severity="info" sx={{ my: 2 }}>
          No improvement plan available at this time.
        </Alert>
      )}
    </Container>
  );
}; 