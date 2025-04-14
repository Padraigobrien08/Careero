import React from 'react';
import { Box, Typography, Paper, Chip } from '@mui/material';
import { ImprovementPlan as ImprovementPlanType } from '../types';

interface ImprovementPlanProps {
  plan: ImprovementPlanType[];
}

export const ImprovementPlan: React.FC<ImprovementPlanProps> = ({ plan }) => {
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Improvement Plan
      </Typography>
      {plan.map((gapPlan, index) => (
        <Paper key={index} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Gap: {gapPlan.gap}
          </Typography>
          {gapPlan.recommendations.map((rec, recIndex) => (
            <Box key={recIndex} sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  {rec.type.toUpperCase()}: {rec.title}
                </Typography>
                <Chip
                  label={rec.priority}
                  color={
                    rec.priority === 'high'
                      ? 'error'
                      : rec.priority === 'medium'
                      ? 'warning'
                      : 'success'
                  }
                  size="small"
                />
              </Box>
              <Typography variant="body2" paragraph>
                {rec.description}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Timeline: {rec.timeline}
              </Typography>
              {rec.resources && rec.resources.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="subtitle2">Resources:</Typography>
                  <ul>
                    {rec.resources.map((resource, resIndex) => (
                      <li key={resIndex}>{resource}</li>
                    ))}
                  </ul>
                </Box>
              )}
            </Box>
          ))}
        </Paper>
      ))}
    </Box>
  );
}; 