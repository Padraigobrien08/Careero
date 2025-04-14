import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import MainLayout from './components/layout/MainLayout';
import Dashboard from './pages/Dashboard';
import ResumeEditor from './pages/ResumeEditor';
import CareerRoadmap from './pages/CareerRoadmap';
import JobMatches from './pages/JobMatches';
import ApplicationTracker from './pages/ApplicationTracker';
import Profile from './pages/Profile';
import Settings from './pages/Settings';
import { v4 as uuidv4 } from 'uuid';

interface Milestone {
  id: string;
  title: string;
  description: string;
  targetDate: string;
  status: 'Not Started' | 'In Progress' | 'Completed';
  priority: 'Low' | 'Medium' | 'High';
  skills: string[];
  resources: string[];
  notes: string;
  skillLevelIncrease?: number;
}

function App() {
  const theme = createTheme({
    palette: {
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#dc004e',
      },
    },
  });

  const [milestones, setMilestones] = useState<Milestone[]>(() => {
    const savedMilestones = localStorage.getItem('careerMilestones');
    return savedMilestones ? JSON.parse(savedMilestones) : [];
  });

  useEffect(() => {
    localStorage.setItem('careerMilestones', JSON.stringify(milestones));
  }, [milestones]);

  const handleAddMilestone = (milestone: Omit<Milestone, 'id'>) => {
    const newMilestone: Milestone = {
      ...milestone,
      id: uuidv4(),
      status: milestone.status || 'Not Started',
      priority: milestone.priority || 'Medium',
      notes: milestone.notes || '',
      skillLevelIncrease: milestone.skillLevelIncrease || 1
    };
    setMilestones(prevMilestones => [...prevMilestones, newMilestone]);
  };

  const handleUpdateMilestone = (updatedMilestone: Milestone) => {
    setMilestones(prevMilestones => 
      prevMilestones.map(milestone => 
        milestone.id === updatedMilestone.id ? updatedMilestone : milestone
      )
    );
  };

  const handleDeleteMilestone = (milestoneId: string) => {
    setMilestones(prevMilestones => 
      prevMilestones.filter(milestone => milestone.id !== milestoneId)
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <MainLayout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/resume" element={<ResumeEditor />} />
              <Route path="/career-roadmap" element={
                <CareerRoadmap 
                  milestones={milestones} 
                  onAddMilestone={handleAddMilestone}
                  onUpdateMilestone={handleUpdateMilestone}
                  onDeleteMilestone={handleDeleteMilestone}
                />
              } />
              <Route path="/job-matches" element={<JobMatches onAddMilestone={handleAddMilestone} />} />
              <Route path="/applications" element={<ApplicationTracker />} />
              <Route path="/profile" element={<Profile />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </MainLayout>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App; 