import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Grid, 
  Button, 
  Divider, 
  Card, 
  CardContent, 
  CardActions,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar 
} from '@mui/material';
import { 
  Description, 
  Work, 
  School, 
  ArrowForward, 
  CalendarToday, 
  Star, 
  StarBorder,
  KeyboardArrowRight,
  BusinessCenter
} from '@mui/icons-material';
import { Link } from 'react-router-dom';

// Add this near the top imports
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Define types
interface Resume {
  name: string;
  email: string;
  phone: string;
  education: Array<{
    institution?: string;
    school?: string;
    degree?: string;
    qualification?: string;
    date?: string;
    duration?: string;
    [key: string]: any;
  }>;
  experience: Array<{
    company?: string;
    organization?: string;
    position?: string;
    title?: string;
    date?: string;
    duration?: string;
    description?: string;
    responsibilities?: string;
    [key: string]: any;
  }>;
  skills: string[];
}

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

interface Skill {
  id: string;
  name: string;
  currentLevel: number;
  targetLevel: number;
}

interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  salary: string;
  description: string;
  requirements: string;
  postedDate: string;
  similarityScore: number;
}

const Dashboard: React.FC = () => {
  const [resume, setResume] = useState<Resume | null>(null);
  const [milestones, setMilestones] = useState<Milestone[]>([]);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [jobMatches, setJobMatches] = useState<Job[]>([]);
  const [loading, setLoading] = useState({
    resume: true,
    milestones: true,
    skills: true,
    jobs: true
  });
  const [error, setError] = useState<string | null>(null);

  // Fetch resume data
  useEffect(() => {
    const fetchResume = async () => {
      try {
        // First try to fetch all resumes
        const resumesResponse = await fetch(`${API_BASE_URL}/resumes`);
        if (resumesResponse.ok) {
          const data = await resumesResponse.json();
          if (data.resumes && data.resumes.length > 0) {
            // Get most recent resume (should be first in list)
            const mostRecentResume = data.resumes[0];
            
            // Then fetch the specific resume content using the filename
            const resumeResponse = await fetch(`${API_BASE_URL}/resumes/${encodeURIComponent(mostRecentResume)}`);
            if (resumeResponse.ok) {
              const resumeData = await resumeResponse.json();
              if (resumeData.parsed_data) {
                // Transform the parsed content to match our expected Resume interface
                const transformedData = {
                  name: "Resume Preview",
                  email: "",
                  phone: "",
                  experience: resumeData.parsed_data.experience || [],
                  education: resumeData.parsed_data.education || [],
                  skills: resumeData.parsed_data.skills || []
                };
                
                setResume(transformedData);
                // Also save to localStorage for easy access later
                localStorage.setItem('dashboardResumeData', JSON.stringify(transformedData));
                setLoading(prev => ({ ...prev, resume: false }));
                return;
              }
            }
          }
        }
        
        // If API fails, try localStorage as fallback
        const savedResume = localStorage.getItem('dashboardResumeData');
        if (savedResume) {
          setResume(JSON.parse(savedResume));
          setLoading(prev => ({ ...prev, resume: false }));
          return;
        }
        
        throw new Error('No resume found');
      } catch (error) {
        console.error('Error fetching resume:', error);
        setError('Could not load resume data');
        setLoading(prev => ({ ...prev, resume: false }));
      }
    };

    fetchResume();
  }, []);

  // Fetch milestones and skills
  useEffect(() => {
    // Get milestones from localStorage
    const savedMilestones = localStorage.getItem('careerMilestones');
    if (savedMilestones) {
      const parsedMilestones = JSON.parse(savedMilestones);
      
      // Sort by target date (closest first)
      const sortedMilestones = [...parsedMilestones].sort((a, b) => {
        return new Date(a.targetDate).getTime() - new Date(b.targetDate).getTime();
      });
      
      setMilestones(sortedMilestones);
    }
    setLoading(prev => ({ ...prev, milestones: false }));

    // Get skills from localStorage
    const savedSkills = localStorage.getItem('careerSkills');
    if (savedSkills) {
      setSkills(JSON.parse(savedSkills));
    }
    setLoading(prev => ({ ...prev, skills: false }));
  }, []);

  // Fetch job matches
  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/jobs`);
        if (!response.ok) {
          throw new Error('Failed to fetch jobs');
        }
        const data = await response.json();
        
        // Sort by similarity score (highest first)
        const sortedJobs = data.sort((a: Job, b: Job) => 
          b.similarityScore - a.similarityScore
        );
        
        setJobMatches(sortedJobs.slice(0, 3)); // Get top 3
      } catch (error) {
        console.error('Error fetching jobs:', error);
        setError('Could not load job matches');
      } finally {
        setLoading(prev => ({ ...prev, jobs: false }));
      }
    };

    fetchJobs();
  }, []);

  // Calculate days until target date
  const getDaysUntil = (targetDate: string): number => {
    const today = new Date();
    const target = new Date(targetDate);
    const diffTime = target.getTime() - today.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays > 0 ? diffDays : 0;
  };

  // Render star rating
  const renderSkillLevel = (currentLevel: number, targetLevel: number) => {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        {[...Array(5)].map((_, i) => (
          <Box key={i}>
            {i < currentLevel ? (
              <Star sx={{ color: 'gold', fontSize: '1.2rem' }} />
            ) : i < targetLevel ? (
              <StarBorder sx={{ color: '#f5b042', fontSize: '1.2rem' }} />
            ) : (
              <StarBorder sx={{ color: 'grey', fontSize: '1.2rem' }} />
            )}
          </Box>
        ))}
      </Box>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid 
        container 
        spacing={3} 
        sx={{ 
          justifyContent: 'center',
          maxWidth: '1400px', 
          mx: 'auto' 
        }}
      >
        {/* Resume Preview */}
        <Grid item xs={12} sm={10} md={5} lg={5}>
          <Paper sx={{ p: 2, height: '100%', maxWidth: '450px', mx: 'auto' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                <Description sx={{ mr: 1, verticalAlign: 'middle' }} />
                Resume Preview
              </Typography>
              <Button 
                component={Link} 
                to="/resume" 
                endIcon={<ArrowForward />}
                size="small"
              >
                Edit Resume
              </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            
            <Box sx={{ maxHeight: 300, overflowY: 'auto', pr: 1 }}>
              {loading.resume ? (
                <Typography>Loading resume...</Typography>
              ) : resume ? (
                <>
                  <Typography variant="h5" align="center" gutterBottom>
                    {resume.name}
                  </Typography>
                  <Typography align="center" gutterBottom>
                    {resume.email} • {resume.phone}
                  </Typography>
                  
                  <Typography variant="subtitle1" sx={{ mt: 2, fontWeight: 'bold' }}>
                    Experience
                  </Typography>
                  <Divider sx={{ mb: 1 }} />
                  {resume.experience?.slice(0, 2).map((exp, index) => (
                    <Box key={index} sx={{ mb: 1 }}>
                      <Typography variant="body1" fontWeight="bold">
                        {exp.position || exp.title || 'Position'} at {exp.company || exp.organization || 'Company'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {exp.date || exp.duration || 'Date'}
                      </Typography>
                      <Typography variant="body2">
                        {(exp.description || exp.responsibilities || '').substring(0, 120)}...
                      </Typography>
                    </Box>
                  ))}
                  
                  <Typography variant="subtitle1" sx={{ mt: 2, fontWeight: 'bold' }}>
                    Education
                  </Typography>
                  <Divider sx={{ mb: 1 }} />
                  {resume.education?.slice(0, 1).map((edu, index) => (
                    <Box key={index} sx={{ mb: 1 }}>
                      <Typography variant="body1" fontWeight="bold">
                        {edu.degree || edu.qualification || 'Degree'}
                      </Typography>
                      <Typography variant="body2">
                        {edu.institution || edu.school || 'Institution'} • {edu.date || edu.duration || 'Date'}
                      </Typography>
                    </Box>
                  ))}
                </>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography color="text.secondary">
                    No resume found. Upload your resume to get started.
                  </Typography>
                  <Button 
                    component={Link} 
                    to="/resume" 
                    variant="contained" 
                    sx={{ mt: 2 }}
                  >
                    Upload Resume
                  </Button>
                </Box>
              )}
            </Box>
          </Paper>
        </Grid>
        
        {/* Upcoming Milestones */}
        <Grid item xs={12} sm={10} md={5} lg={5}>
          <Paper sx={{ p: 2, height: '100%', maxWidth: '450px', mx: 'auto' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                <CalendarToday sx={{ mr: 1, verticalAlign: 'middle' }} />
                Upcoming Career Milestones
              </Typography>
              <Button 
                component={Link} 
                to="/career-roadmap" 
                endIcon={<ArrowForward />}
                size="small"
              >
                View All
              </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            
            {loading.milestones ? (
              <Typography>Loading milestones...</Typography>
            ) : milestones.length > 0 ? (
              <Box>
                {milestones
                  .filter(m => m.status !== 'Completed')
                  .slice(0, 3)
                  .map((milestone) => {
                    const daysRemaining = getDaysUntil(milestone.targetDate);
                    const progress = milestone.status === 'In Progress' ? 50 : 
                                   milestone.status === 'Completed' ? 100 : 0;
                    
                    return (
                      <Card key={milestone.id} sx={{ mb: 2 }} variant="outlined">
                        <CardContent sx={{ pb: 1 }}>
                          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            {milestone.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" noWrap>
                            {milestone.description}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, mb: 1 }}>
                            <Box sx={{ flexGrow: 1, mr: 1 }}>
                              <LinearProgress 
                                variant="determinate" 
                                value={progress} 
                                sx={{ height: 8, borderRadius: 4 }}
                              />
                            </Box>
                            <Typography variant="caption" sx={{ minWidth: 60 }}>
                              {progress}% Complete
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Chip 
                              label={`${daysRemaining} days remaining`}
                              color={daysRemaining < 7 ? 'error' : daysRemaining < 14 ? 'warning' : 'success'}
                              size="small"
                            />
                            <Typography variant="caption" color="text.secondary">
                              Due: {new Date(milestone.targetDate).toLocaleDateString()}
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    );
                  })}
                </Box>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography color="text.secondary">
                  No milestones found. Add career milestones to track your progress.
                </Typography>
                <Button 
                  component={Link} 
                  to="/career-roadmap" 
                  variant="contained" 
                  sx={{ mt: 2 }}
                >
                  Add Milestones
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>
        
        {/* Skills Development */}
        <Grid item xs={12} sm={10} md={5} lg={5}>
          <Paper sx={{ p: 2, height: '100%', maxWidth: '450px', mx: 'auto' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                <School sx={{ mr: 1, verticalAlign: 'middle' }} />
                Skills Development
              </Typography>
              <Button 
                component={Link} 
                to="/career-roadmap" 
                endIcon={<ArrowForward />}
                size="small"
              >
                Manage Skills
              </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            
            {loading.skills ? (
              <Typography>Loading skills...</Typography>
            ) : skills.length > 0 ? (
              <List>
                {skills.map((skill) => (
                  <ListItem key={skill.id}>
                    <ListItemText
                      primary={skill.name}
                      secondary={
                        <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                          {renderSkillLevel(skill.currentLevel, skill.targetLevel)}
                          <Typography variant="caption" sx={{ ml: 1 }}>
                            Level {skill.currentLevel}/{skill.targetLevel}
                          </Typography>
                        </Box>
                      }
                    />
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Chip 
                        label={
                          skill.currentLevel >= skill.targetLevel 
                            ? 'Achieved' 
                            : `${skill.targetLevel - skill.currentLevel} to go`
                        }
                        color={
                          skill.currentLevel >= skill.targetLevel 
                            ? 'success' 
                            : skill.currentLevel >= skill.targetLevel - 1 
                              ? 'warning' 
                              : 'primary'
                        }
                        size="small"
                      />
                    </Box>
                  </ListItem>
                ))}
              </List>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography color="text.secondary">
                  No skills found. Add skills to track your development.
                </Typography>
                <Button 
                  component={Link} 
                  to="/career-roadmap" 
                  variant="contained" 
                  sx={{ mt: 2 }}
                >
                  Add Skills
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>
        
        {/* Job Matches */}
        <Grid item xs={12} sm={10} md={5} lg={5}>
          <Paper sx={{ p: 2, height: '100%', maxWidth: '450px', mx: 'auto' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                <BusinessCenter sx={{ mr: 1, verticalAlign: 'middle' }} />
                Top Job Matches
              </Typography>
              <Button 
                component={Link} 
                to="/job-matches" 
                endIcon={<ArrowForward />}
                size="small"
              >
                View All Matches
              </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            
            {loading.jobs ? (
              <Typography>Loading job matches...</Typography>
            ) : jobMatches.length > 0 ? (
              <List>
                {jobMatches.map((job) => (
                  <ListItem 
                    key={job.id}
                    secondaryAction={
                      <Button 
                        component={Link}
                        to={`/job-matches?job=${job.id}`}
                        endIcon={<KeyboardArrowRight />}
                        size="small"
                      >
                        Details
                      </Button>
                    }
                    sx={{ 
                      mb: 1,
                      p: 1.5,
                      borderRadius: 1,
                      border: '1px solid',
                      borderColor: 'divider',
                      '&:hover': { bgcolor: 'action.hover' }
                    }}
                  >
                    <ListItemIcon>
                      <Avatar sx={{ bgcolor: 'primary.main' }}>
                        {job.company.charAt(0)}
                      </Avatar>
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            {job.title}
                          </Typography>
                          <Chip 
                            label={`${Math.round(job.similarityScore * 100)}% Match`}
                            color="primary"
                            size="small"
                            sx={{ ml: 1 }}
                          />
                        </Box>
                      }
                      secondary={
                        <>
                          <Typography variant="body2" component="span">
                            {job.company} • {job.location}
                          </Typography>
                          <br />
                          <Typography variant="caption" color="text.secondary" component="span">
                            Posted: {new Date(job.postedDate).toLocaleDateString()}
                          </Typography>
                        </>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography color="text.secondary">
                  No job matches found. Upload your resume to get personalized job matches.
                </Typography>
                <Button 
                  component={Link} 
                  to="/job-matches" 
                  variant="contained" 
                  sx={{ mt: 2 }}
                >
                  Find Job Matches
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 