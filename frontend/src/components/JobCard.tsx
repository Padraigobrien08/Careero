import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { Job } from '../types';

interface JobCardProps {
  job: Job;
  showScore?: boolean;
}

export const JobCard: React.FC<JobCardProps> = ({ job, showScore = false }) => {
  return (
    <Card sx={{ mb: 2, '&:hover': { boxShadow: 6 } }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box>
            <Typography variant="h5" component="div">
              {job.title}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {job.company} • {job.location}
            </Typography>
            {job.salary && (
              <Typography variant="body2" color="text.secondary">
                Salary: {job.salary}
              </Typography>
            )}
          </Box>
          {showScore && (
            <Chip
              label={`${(job.similarityScore * 100).toFixed(1)}% Match`}
              color="primary"
              variant="outlined"
            />
          )}
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-line' }}>
          {job.description}
        </Typography>
        {job.requirements && job.requirements.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Requirements:
            </Typography>
            <ul style={{ margin: 0, paddingLeft: '20px' }}>
              {job.requirements.map((req, index) => (
                <li key={index}>
                  <Typography variant="body2">{req}</Typography>
                </li>
              ))}
            </ul>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}; 