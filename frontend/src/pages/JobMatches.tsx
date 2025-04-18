import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  CircularProgress,
  Alert,
  Chip,
  Divider,
  TextField,
  Grid,
  Tabs,
  Tab,
  ListItemIcon,
  Card,
  CardContent,
  CardActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  Slider,
} from '@mui/material';
import {
  Visibility,
  Close,
  Edit,
  Download,
  Add as AddIcon,
} from '@mui/icons-material';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Cache for job matches
let cachedJobs: Job[] = []; // Initialize as empty array instead of null
let lastFetchTime: number | null = null;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes
const MAX_JOBS = 25; // Maximum number of jobs to display
const SEARCH_DEBOUNCE = 500; // 500ms debounce time

interface Job {
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

interface AddJobDialogProps {
  open: boolean;
  onClose: () => void;
  onAdd: (job: Omit<Job, 'id' | 'postedDate' | 'similarityScore'>) => Promise<void>;
}

interface ResumeTailoring {
  specific_edits: string[];
  sections_to_focus: string[];
  keywords: string[];
  skills_to_emphasize: string[];
  experience_to_highlight: string[];
}

interface RoadmapMilestone {
  id: string;
  title: string;
  description: string;
  targetDate: string;
  status: 'Not Started' | 'In Progress' | 'Completed';
  priority: 'Low' | 'Medium' | 'High';
  skills: string[];
  resources: string[];
  notes: string;
}

interface Skill {
  id: string;
  name: string;
  currentLevel: number;
  targetLevel: number;
}

interface JobMatchesProps {
  onAddMilestone: (milestone: any) => void;
}

// Fetches the text of the most recent resume from the backend
async function getCurrentResumeText(): Promise<string | null> {
  console.log("Attempting to retrieve most recent resume text from backend...");
  try {
    // 1. Fetch the list of resume filenames
    const listResponse = await fetch(`${API_BASE_URL}/resumes`);
    if (!listResponse.ok) {
      console.error("Failed to fetch resume list:", listResponse.status);
      return null;
    }
    const listData = await listResponse.json();
    console.log("[getCurrentResumeText] Raw list data from /resumes:", JSON.stringify(listData)); // DEBUG LOG
    
    // Check if the list exists and has filenames
    if (!listData || !Array.isArray(listData.resumes) || listData.resumes.length === 0) {
      console.warn("[getCurrentResumeText] No resumes found or invalid format in listData.resumes.");
      return null;
    }
    
    // 2. Get the filename of the most recent resume (assuming first in list)
    const resumeFilename = listData.resumes[0]; 
    console.log("[getCurrentResumeText] Extracted filename (listData.resumes[0]):", resumeFilename); // DEBUG LOG

    if (!resumeFilename || typeof resumeFilename !== 'string') {
        console.error("[getCurrentResumeText] Could not identify a valid string filename. Value was:", resumeFilename);
        return null;
    }
    console.log(`[getCurrentResumeText] Proceeding to fetch content for filename: ${resumeFilename}`);

    // 3. Fetch the specific resume content using the filename
    const encodedFilename = encodeURIComponent(resumeFilename);
    console.log(`[getCurrentResumeText] Fetching: /resumes/${encodedFilename}`); // DEBUG LOG
    const contentResponse = await fetch(`${API_BASE_URL}/resumes/${encodedFilename}`); // Ensure filename is URL encoded
    if (!contentResponse.ok) {
        console.error(`Failed to fetch content for resume ${resumeFilename}:`, contentResponse.status);
        return null;
    }
    const contentData = await contentResponse.json();

    // 4. Extract the parsed text
    // Assuming backend returns { parsed_data: { text: "..." } }
    if (contentData && contentData.parsed_data && typeof contentData.parsed_data.text === 'string') {
      console.log("Successfully retrieved resume text from backend.");
      return contentData.parsed_data.text;
    } else {
      console.warn("Parsed resume text not found in the backend response for filename:", resumeFilename, "Response structure:", contentData);
      return null;
    }

  } catch (error) {
    console.error("Error retrieving resume text:", error);
    return null;
  }
}

const AddJobDialog: React.FC<AddJobDialogProps> = ({ open, onClose, onAdd }) => {
  const [formData, setFormData] = useState({
    title: '',
    company: '',
    location: '',
    salary: '',
    description: '',
    requirements: '',
  });
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    try {
      if (!formData.title || !formData.description) {
        setError('Title and description are required');
        return;
      }

      await onAdd({
        ...formData,
        requirements: formData.requirements.split('\n').filter(req => req.trim()),
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add job');
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Add a Watched Job</DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Job Title"
              value={formData.title}
              onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              required
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Company"
              value={formData.company}
              onChange={(e) => setFormData({ ...formData, company: e.target.value })}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Location"
              value={formData.location}
              onChange={(e) => setFormData({ ...formData, location: e.target.value })}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Salary"
              value={formData.salary}
              onChange={(e) => setFormData({ ...formData, salary: e.target.value })}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Job Description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              multiline
              rows={4}
              required
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Requirements (one per line)"
              value={formData.requirements}
              onChange={(e) => setFormData({ ...formData, requirements: e.target.value })}
              multiline
              rows={4}
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained" color="primary">
          Add Job
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const JobMatches: React.FC<JobMatchesProps> = ({ onAddMilestone }) => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [addJobDialogOpen, setAddJobDialogOpen] = useState(false);
  const [newlyAddedJob, setNewlyAddedJob] = useState<Job | null>(null);
  const [showNewJob, setShowNewJob] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [searchTimeout, setSearchTimeout] = useState<NodeJS.Timeout | null>(null);
  const [tailoringSuggestions, setTailoringSuggestions] = useState<ResumeTailoring | null>(null);
  const [coverLetter, setCoverLetter] = useState<string | null>(null);
  const [roadmap, setRoadmap] = useState<RoadmapMilestone[]>([]);
  const [selectedMilestone, setSelectedMilestone] = useState<RoadmapMilestone | null>(null);
  const [milestoneDialogOpen, setMilestoneDialogOpen] = useState(false);
  const [newMilestone, setNewMilestone] = useState<RoadmapMilestone>({
    id: '',
    title: '',
    description: '',
    targetDate: '',
    status: 'Not Started',
    priority: 'Low',
    skills: [],
    resources: [],
    notes: '',
  });
  const [activeTab, setActiveTab] = useState(0);
  const [loadingFeature, setLoadingFeature] = useState<string | null>(null);
  const [jobScores, setJobScores] = useState<Record<string, number>>({});
  const [skills, setSkills] = useState<Skill[]>([]);
  const [selectedSkillId, setSelectedSkillId] = useState<string>('');
  const [skillLevelIncrease, setSkillLevelIncrease] = useState<number>(1);
  const [tailoredResume, setTailoredResume] = useState<{ sections: Array<{ title: string, content: string }> } | null>(null);

  useEffect(() => {
    fetchJobs();
  }, []);

  useEffect(() => {
    const savedSkills = localStorage.getItem('careerSkills');
    if (savedSkills) {
      setSkills(JSON.parse(savedSkills));
    }
  }, []);

  const fetchJobs = async () => {
    try {
      setLoading(true);
      setError(null);
      setIsSearchMode(false);

      // Check if we have cached data that's still valid
      const now = Date.now();
      if (cachedJobs.length > 0 && lastFetchTime && (now - lastFetchTime) < CACHE_DURATION) {
        setJobs(cachedJobs.slice(0, MAX_JOBS));
        setLoading(false);
        return;
      }

      const response = await fetch(`${API_BASE_URL}/jobs`);
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Please upload a resume first to see job matches');
        }
        throw new Error('Failed to fetch jobs');
      }
      const data = await response.json();
      
      // Update cache and ensure we only keep MAX_JOBS
      cachedJobs = Array.isArray(data) ? data.slice(0, MAX_JOBS) : [];
      lastFetchTime = now;
      
      setJobs(cachedJobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setJobs([]); // Set empty array on error
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      fetchJobs(); // Reset to top matches if search is empty
      return;
    }

    try {
      setIsSearching(true);
      setError(null);
      setIsSearchMode(true);
      const response = await fetch(`${API_BASE_URL}/jobs/search?query=${encodeURIComponent(searchQuery)}`);
      if (!response.ok) {
        throw new Error('Failed to search jobs');
      }
      const data = await response.json();
      setJobs(data.jobs.slice(0, MAX_JOBS)); // Limit search results too
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setJobs([]); // Set empty array on error
    } finally {
      setIsSearching(false);
    }
  }, [searchQuery, fetchJobs]);

  const debouncedSearch = useCallback(() => {
    if (searchTimeout) {
      clearTimeout(searchTimeout);
    }

    const timeout = setTimeout(() => {
      handleSearch();
    }, SEARCH_DEBOUNCE);

    setSearchTimeout(timeout);
  }, [handleSearch, searchTimeout]);

  useEffect(() => {
    return () => {
      if (searchTimeout) {
        clearTimeout(searchTimeout);
      }
    };
  }, [searchTimeout]);

  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    debouncedSearch();
  };

  const handleViewJob = async (job: Job) => {
    if (!job.id) {
      setError('Invalid job ID');
      return;
    }
    
    setSelectedJob(job);
    setOpenDialog(true);
    
    // Calculate match score if not already calculated
    if (!jobScores[job.id]) {
      try {
        // Call the placeholder function to get resume text
        const currentResumeText = await getCurrentResumeText(); 
        
        if (!currentResumeText) { // Check if text was retrieved
          console.warn("Resume text not available, skipping match score calculation.");
          setJobScores(prev => ({ ...prev, [job.id]: -1 })); // Indicate missing resume
          return; // Don't make the fetch call if no resume text
        }

        // Proceed with the fetch call using the retrieved text
        const response = await fetch(`${API_BASE_URL}/jobs/${job.id}/match`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            // Add other headers like Authorization if needed
          },
          body: JSON.stringify({ resume_text: currentResumeText })
        });

        if (!response.ok) {
           const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error response' }));
           throw new Error(errorData.detail || `Failed to calculate match score: ${response.status}`);
        }
        const data = await response.json();
        setJobScores(prev => ({
          ...prev,
          [job.id]: data.similarity_score // Ensure backend sends 'similarity_score'
        }));
      } catch (err) {
        console.error('Error calculating match score:', err);
        setJobScores(prev => ({ ...prev, [job.id]: -1 })); // Indicate error
      }
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch (e) {
      return 'Unknown date';
    }
  };

  const handleAddJob = async (jobData: Omit<Job, 'id' | 'postedDate' | 'similarityScore'>) => {
    try {
      const response = await fetch(`${API_BASE_URL}/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(jobData),
      });

      if (!response.ok) {
        throw new Error('Failed to add job');
      }

      const result = await response.json();
      console.log("[handleAddJob] Received result:", result); // Add log to see the structure

      // Create a new job object with the response data (access directly from result)
      const newJob: Job = { 
        id: result.id, // Access result.id directly
        title: result.title, // Access result.title directly
        company: result.company, // etc.
        location: result.location,
        salary: result.salary,
        description: result.description,
        requirements: Array.isArray(result.requirements) ? result.requirements : [],
        postedDate: result.postedDate,
        similarityScore: result.similarityScore ?? 0.0 // Use nullish coalescing for default
      };

      // Set the newly added job for the highlighted section
      setNewlyAddedJob(newJob);
      setShowNewJob(true);
      
      // Refresh the entire job list to ensure consistent matching
      await fetchJobs();
    } catch (err) {
      throw new Error('Failed to add job');
    }
  };

  const handleTailorResume = async (jobId: string) => {
    try {
      setLoadingFeature('resume');
      setError(null); // Clear previous errors

      // Fetch resume text first
      const currentResumeText = await getCurrentResumeText();
      if (!currentResumeText) {
        throw new Error('Could not retrieve resume text. Please ensure a resume is uploaded.');
      }

      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/tailor-resume`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ resume_text: currentResumeText })
      });

      if (!response.ok) {
        // Use status code for more specific error messages if needed
        const errorData = await response.json().catch(() => ({ detail: 'Failed to tailor resume' }));
        throw new Error(errorData.detail || `Failed to tailor resume: ${response.status}`);
      }

      const data = await response.json();
      console.log("!!! handleTailorResume: Data Received !!!", data); // Simplified log
      setTailoredResume(data); // Assuming backend returns the tailored resume structure
      setSelectedJob(jobs.find(job => job.id === jobId) || null);
      setActiveTab(1); // Switch to the Tailored Resume tab
    } catch (error) {
      console.error('Error tailoring resume:', error);
      setError(error instanceof Error ? error.message : 'Failed to tailor resume');
    } finally {
      setLoadingFeature(null);
    }
  };

  const handleGenerateCoverLetter = async (jobId: string) => {
    try {
      setLoadingFeature('coverLetter');
      setError(null); // Clear previous errors

      // Fetch resume text first
      const currentResumeText = await getCurrentResumeText();
      if (!currentResumeText) {
        throw new Error('Could not retrieve resume text. Please ensure a resume is uploaded.');
      }

      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/generate-cover-letter`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ resume_text: currentResumeText })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to generate cover letter' }));
        throw new Error(errorData.detail || `Failed to generate cover letter: ${response.status}`);
      }

      const data = await response.json();
      setCoverLetter(data.cover_letter); // Assuming backend returns { cover_letter: "..." }
      setSelectedJob(jobs.find(job => job.id === jobId) || null);
      setActiveTab(2); // Switch to the Cover Letter tab
    } catch (error) {
      console.error('Error generating cover letter:', error);
      setError(error instanceof Error ? error.message : 'Failed to generate cover letter');
    } finally {
      setLoadingFeature(null);
    }
  };

  const handleGenerateRoadmap = async (jobId: string) => {
    try {
      setLoadingFeature('roadmap');
      setError(null); // Clear previous errors

      // Fetch resume text first
      const currentResumeText = await getCurrentResumeText();
      if (!currentResumeText) {
        throw new Error('Could not retrieve resume text. Please ensure a resume is uploaded.');
      }

      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/generate-roadmap`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ resume_text: currentResumeText })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to generate roadmap' }));
        throw new Error(errorData.detail || `Failed to generate roadmap: ${response.status}`);
      }

      const data = await response.json();
      // Assuming backend returns { roadmap: [...] }
      setRoadmap(Array.isArray(data.roadmap) ? data.roadmap : []); 
      setSelectedJob(jobs.find(job => job.id === jobId) || null);
      setActiveTab(3); // Switch to the Roadmap tab
    } catch (error) {
      console.error('Error generating roadmap:', error);
      setError(error instanceof Error ? error.message : 'Failed to generate roadmap');
    } finally {
      setLoadingFeature(null);
    }
  };

  const handleDownloadCoverLetter = () => {
    if (!coverLetter) return;
    const element = document.createElement('a');
    const file = new Blob([coverLetter], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `cover_letter_${selectedJob?.company}_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleAddToRoadmap = (item: Job | RoadmapMilestone, isRoadmapItem: boolean = false) => {
    // Don't set the job itself as selectedMilestone since they have different types
    // Just store the ID to know which item we're working with
    const itemId = isRoadmapItem ? (item as RoadmapMilestone).id : (item as Job).id;
    
    // Set default skill if available
    const defaultSkillId = skills.length > 0 ? skills[0].id : '';
    setSelectedSkillId(defaultSkillId);
    setSkillLevelIncrease(1);
    
    if (isRoadmapItem) {
      // If it's a roadmap milestone, use its data
      const milestone = item as RoadmapMilestone;
      setNewMilestone({
        id: '',
        title: milestone.title,
        description: milestone.description,
        targetDate: milestone.targetDate || new Date().toISOString().split('T')[0],
        status: 'Not Started',
        priority: 'Medium',
        skills: defaultSkillId ? [defaultSkillId] : [],
        resources: milestone.resources || [],
        notes: '',
      });
    } else {
      // If it's a job, use job data
      const job = item as Job;
      setNewMilestone({
        id: '',
        title: job.title,
        description: job.description,
        targetDate: new Date().toISOString().split('T')[0],
        status: 'Not Started',
        priority: 'Medium',
        skills: defaultSkillId ? [defaultSkillId] : [],
        resources: [],
        notes: '',
      });
    }
    
    setMilestoneDialogOpen(true);
  };

  const handleSaveMilestone = () => {
    // Update the milestone with the selected skill and other data
    const milestoneToAdd = {
      ...newMilestone,
      skills: selectedSkillId ? [selectedSkillId] : [],
      skillLevelIncrease: skillLevelIncrease // Add this to pass the level increase information
    };
    
    // Pass the milestone to the parent component
    onAddMilestone(milestoneToAdd);
    setMilestoneDialogOpen(false);
    
    // Show success message
    alert("Milestone added to Career Roadmap!");
  };

  const handleCloseJobDialog = () => {
    setOpenDialog(false);
    setSelectedJob(null);
    setActiveTab(0);
    setTailoredResume(null);
    setCoverLetter(null);
    setRoadmap([]);
  };

  return (
    <Box sx={{ p: 3, width: '100%', maxWidth: 'none' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          {isSearchMode ? 'Search Results' : 'Top 25 Job Matches'}
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setAddJobDialogOpen(true)}
          sx={{ px: 2, minWidth: 'auto', whiteSpace: 'nowrap' }}
        >
          Add a Job
        </Button>
      </Box>

      <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search for specific jobs..."
          value={searchQuery}
          onChange={handleSearchInputChange}
        />
        {isSearchMode && (
          <Button
            variant="outlined"
            onClick={() => {
              setSearchQuery('');
              fetchJobs();
            }}
          >
            Back to Top Matches
          </Button>
        )}
      </Box>

      {error && (
        <Alert 
          severity={error.includes('upload a resume') ? 'info' : 'error'} 
          sx={{ mb: 3 }}
          action={
            error.includes('upload a resume') ? (
              <Button 
                color="inherit" 
                size="small"
                onClick={() => window.location.href = '/resume-editor'}
              >
                Upload Resume
              </Button>
            ) : null
          }
        >
          {error}
        </Alert>
      )}

      {showNewJob && newlyAddedJob && (
        <Paper sx={{ mb: 3, p: 2, backgroundColor: '#e3f2fd' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography variant="h6">{newlyAddedJob.title}</Typography>
              <Typography variant="body2" color="text.secondary">
                {newlyAddedJob.company} • {newlyAddedJob.location}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Match Score: {(newlyAddedJob.similarityScore * 100).toFixed(1)}%
              </Typography>
            </Box>
            <Button 
              variant="outlined" 
              onClick={() => setShowNewJob(false)}
              startIcon={<Close />}
            >
              Close
            </Button>
          </Box>
        </Paper>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
          <CircularProgress />
        </Box>
      ) : jobs.length > 0 ? (
        <Paper>
          <List>
            {jobs.map((job) => (
              <React.Fragment key={job.id}>
                <ListItem>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="h6">{job.title}</Typography>
                        {!isSearchMode && jobScores[job.id] !== undefined && (
                          <Chip 
                            label={`${(jobScores[job.id] * 100).toFixed(1)}% Match`}
                            color="primary"
                            size="small"
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          {job.company} • {job.location}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Posted: {formatDate(job.postedDate)}
                        </Typography>
                        {job.salary && (
                          <Typography variant="body2" color="text.secondary">
                            Salary: {job.salary}
                          </Typography>
                        )}
                        {jobScores[job.id] !== undefined && (
                          <Typography variant="body2" color="text.secondary">
                            Match Score: {(jobScores[job.id] * 100).toFixed(1)}%
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={() => handleViewJob(job)}>
                      <Visibility />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider />
              </React.Fragment>
            ))}
          </List>
        </Paper>
      ) : !error && (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            {isSearchMode 
              ? 'No jobs found matching your search. Try different keywords.' 
              : 'No job matches found. Upload a resume to see matching jobs.'}
          </Typography>
          {!isSearchMode && (
            <Button 
              variant="contained" 
              sx={{ mt: 2 }}
              onClick={() => window.location.href = '/resume-editor'}
            >
              Upload Resume
            </Button>
          )}
        </Paper>
      )}

      {selectedJob && roadmap.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Development Roadmap for {selectedJob.title}
          </Typography>
          <Grid container spacing={3}>
            {roadmap.map((milestone, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {milestone.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {milestone.description}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Timeframe: {milestone.targetDate}
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Skills to Develop:
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {milestone.skills.map((skill, i) => (
                          <Chip key={i} label={skill} size="small" />
                        ))}
                      </Box>
                    </Box>
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Resources:
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {milestone.resources.map((resource, i) => (
                          <Chip key={i} label={resource} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  </CardContent>
                  <CardActions>
                    <Button
                      startIcon={<AddIcon />}
                      onClick={() => handleAddToRoadmap(milestone, true)}
                    >
                      Add to Career Roadmap
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Job Details Dialog */}
      <Dialog 
        open={openDialog} 
        onClose={handleCloseJobDialog} 
        maxWidth="md" 
        fullWidth
        PaperProps={{
          sx: { maxHeight: "90vh", overflowY: "auto" }
        }}
      >
        {selectedJob ? (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h5">{selectedJob.title}</Typography>
                <IconButton onClick={handleCloseJobDialog} edge="end" aria-label="close">
                  <Close />
                </IconButton>
              </Box>
              <Typography variant="subtitle1" color="text.secondary">
                {selectedJob.company} {selectedJob.location ? `• ${selectedJob.location}` : ''}
              </Typography>
            </DialogTitle>
            
            {/* Tabs for different content */}
            <Tabs
              value={activeTab}
              onChange={(_, newValue) => setActiveTab(newValue)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ px: 2, borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab label="Job Details" />
              {tailoredResume && <Tab label="Tailored Resume" />}
              {coverLetter && <Tab label="Cover Letter" />}
              {roadmap && roadmap.length > 0 && <Tab label="Career Roadmap" />}
            </Tabs>
            
            <DialogContent dividers>
              {/* Job Details Tab */}
              {activeTab === 0 && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="h6" gutterBottom>Job Description</Typography>
                  <Typography paragraph>{selectedJob.description}</Typography>
                  
                  <Typography variant="h6" gutterBottom>Requirements</Typography>
                  <List>
                    {Array.isArray(selectedJob.requirements)
                      ? selectedJob.requirements.map((req, index) => (
                          <ListItem key={index} sx={{ py: 0 }}>
                            <Typography>• {req}</Typography>
                          </ListItem>
                        ))
                      : <ListItem><Typography variant="body2" color="text.secondary">No requirements listed.</Typography></ListItem> /* Handle non-array case */
                    }
                  </List>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Posted: {formatDate(selectedJob.postedDate)}
                    </Typography>
                    {selectedJob.salary && (
                      <Typography variant="body2" color="text.secondary">
                        Salary: {selectedJob.salary}
                      </Typography>
                    )}
                    <Typography variant="body2" color="text.secondary">
                      Match Score: {(selectedJob.similarityScore * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>
              )}
              
              {/* Tailored Resume Tab */}
              {activeTab === 1 && tailoredResume && (
                <Box sx={{ mt: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
                    <Button 
                      variant="outlined" 
                      startIcon={<Download />}
                      onClick={() => {
                        // Format the object into a readable string for copying
                        let copyText = "";
                        for (const [key, value] of Object.entries(tailoredResume)) {
                          if (Array.isArray(value)) {
                            copyText += `**${key.replace(/_/g, ' ').toUpperCase()}**\n`;
                            copyText += value.join('\n') + '\n\n';
                          }
                        }
                        navigator.clipboard.writeText(copyText.trim());
                        alert('Tailored resume suggestions copied to clipboard!');
                      }}
                    >
                      Copy Suggestions
                    </Button>
                  </Box>
                  
                  {/* Iterate through the properties of the tailoredResume object */}
                  {Object.entries(tailoredResume).map(([key, value]) => {
                    // Check if the value is an array before trying to map it
                    if (Array.isArray(value)) {
                      return (
                        <Box key={key} sx={{ mb: 3 }}>
                          {/* Format the key as a heading */}
                          <Typography variant="h6" gutterBottom>
                            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </Typography>
                          <List dense>
                            {value.map((item, index) => (
                              <ListItem key={index} sx={{ py: 0.5 }}>
                                <Typography variant="body1">• {String(item)}</Typography>
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      );
                    } else {
                       // Optionally render non-array properties if needed
                       // console.log("Skipping non-array property:", key);
                       return null;
                    }
                  })}
                </Box>
              )}
              
              {/* Cover Letter Tab */}
              {activeTab === 2 && coverLetter && (
                <Box sx={{ mt: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
                    <Button 
                      variant="outlined" 
                      startIcon={<Download />}
                      onClick={() => {
                        navigator.clipboard.writeText(coverLetter);
                        alert('Cover letter copied to clipboard!');
                      }}
                    >
                      Copy to Clipboard
                    </Button>
                  </Box>
                  <Typography paragraph style={{ whiteSpace: 'pre-line' }}>
                    {coverLetter}
                  </Typography>
                </Box>
              )}
              
              {/* Roadmap Tab */}
              {activeTab === 3 && roadmap && roadmap.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="h6" gutterBottom>
                    Development Roadmap for {selectedJob.title}
                  </Typography>
                  <Grid container spacing={3}>
                    {roadmap.map((milestone, index) => (
                      <Grid item xs={12} sm={6} key={index}>
                        <Card>
                          <CardContent>
                            <Typography variant="h6" gutterBottom>
                              {milestone.title}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                              {milestone.description}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                              Timeframe: {milestone.targetDate}
                            </Typography>
                            <Box sx={{ mb: 2 }}>
                              <Typography variant="subtitle2" gutterBottom>
                                Skills to Develop:
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {milestone.skills.map((skill, i) => (
                                  <Chip key={i} label={skill} size="small" />
                                ))}
                              </Box>
                            </Box>
                            <Box>
                              <Typography variant="subtitle2" gutterBottom>
                                Resources:
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {milestone.resources.map((resource, i) => (
                                  <Chip key={i} label={resource} size="small" variant="outlined" />
                                ))}
                              </Box>
                            </Box>
                          </CardContent>
                          <CardActions>
                            <Button
                              startIcon={<AddIcon />}
                              onClick={() => handleAddToRoadmap(milestone, true)}
                            >
                              Add to Career Roadmap
                            </Button>
                          </CardActions>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}
              
              {/* Loading State */}
              {loadingFeature && (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
                  <CircularProgress />
                </Box>
              )}
            </DialogContent>
            
            <DialogActions>
              {activeTab === 0 && (
                <>
                  <Button 
                    onClick={() => handleTailorResume(selectedJob.id)}
                    color="primary"
                    variant="outlined"
                    disabled={loadingFeature === 'resume'}
                    startIcon={loadingFeature === 'resume' ? <CircularProgress size={20} /> : <Edit />}
                  >
                    Tailor My Resume
                  </Button>
                  <Button 
                    onClick={() => handleGenerateCoverLetter(selectedJob.id)}
                    color="primary" 
                    variant="outlined"
                    disabled={loadingFeature === 'coverLetter'}
                    startIcon={loadingFeature === 'coverLetter' ? <CircularProgress size={20} /> : <Edit />}
                  >
                    Generate Cover Letter
                  </Button>
                  <Button 
                    onClick={() => handleGenerateRoadmap(selectedJob.id)}
                    color="primary" 
                    variant="outlined"
                    disabled={loadingFeature === 'roadmap'}
                    startIcon={loadingFeature === 'roadmap' ? <CircularProgress size={20} /> : <Edit />}
                  >
                    Generate Roadmap
                  </Button>
                </>
              )}
              <Button 
                onClick={handleCloseJobDialog} 
                color="primary" 
                variant="contained"
              >
                Close
              </Button>
            </DialogActions>
          </>
        ) : (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h5">Job Details</Typography>
                <IconButton onClick={handleCloseJobDialog} edge="end" aria-label="close">
                  <Close />
                </IconButton>
              </Box>
            </DialogTitle>
            <DialogContent dividers>
              <Typography>Loading job details...</Typography>
            </DialogContent>
            <DialogActions>
              <Button 
                onClick={handleCloseJobDialog} 
                color="primary" 
                variant="contained"
              >
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      <AddJobDialog
        open={addJobDialogOpen}
        onClose={() => setAddJobDialogOpen(false)}
        onAdd={handleAddJob}
      />

      <Dialog open={milestoneDialogOpen} onClose={() => setMilestoneDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Milestone to Career Roadmap</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              label="Title"
              value={newMilestone.title}
              onChange={(e) => setNewMilestone({ ...newMilestone, title: e.target.value })}
              fullWidth
            />
            <TextField
              label="Description"
              value={newMilestone.description}
              onChange={(e) => setNewMilestone({ ...newMilestone, description: e.target.value })}
              multiline
              rows={4}
              fullWidth
            />
            <TextField
              label="Target Date"
              type="date"
              value={newMilestone.targetDate}
              onChange={(e) => setNewMilestone({ ...newMilestone, targetDate: e.target.value })}
              InputLabelProps={{ shrink: true }}
              fullWidth
            />
            
            {/* Skill Selection */}
            <FormControl fullWidth>
              <InputLabel>Associated Skill</InputLabel>
              <Select
                value={selectedSkillId}
                onChange={(e) => {
                  setSelectedSkillId(e.target.value);
                  setNewMilestone({
                    ...newMilestone,
                    skills: e.target.value ? [e.target.value] : []
                  });
                }}
                label="Associated Skill"
              >
                {skills.map((skill) => (
                  <MenuItem key={skill.id} value={skill.id}>
                    {skill.name} (Current Level: {skill.currentLevel}/{skill.targetLevel})
                  </MenuItem>
                ))}
                {skills.length === 0 && (
                  <MenuItem disabled value="">
                    No skills available. Add skills in Career Roadmap first.
                  </MenuItem>
                )}
              </Select>
            </FormControl>
            
            {/* Skill Level Increase */}
            {selectedSkillId && (
              <Box>
                <Typography gutterBottom>
                  Skill Level Increase (when completed)
                </Typography>
                <Slider
                  value={skillLevelIncrease}
                  onChange={(_, value) => setSkillLevelIncrease(value as number)}
                  min={1}
                  max={5}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                />
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMilestoneDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleSaveMilestone} 
            variant="contained" 
            color="primary"
            disabled={!selectedSkillId}
          >
            Add Milestone
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default JobMatches; 