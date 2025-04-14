import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Card,
  CardContent,
  CardActions,
  Divider,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Add, Edit, Delete, Star } from '@mui/icons-material';
import { v4 as uuidv4 } from 'uuid';

interface Skill {
  id: string;
  name: string;
  currentLevel: number;
  targetLevel: number;
}

interface CareerMilestone {
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

interface CareerRoadmapContentProps {
  milestones: CareerMilestone[];
  onUpdateMilestone: (milestone: CareerMilestone) => void;
}

const CareerRoadmapContent: React.FC<CareerRoadmapContentProps> = ({ milestones, onUpdateMilestone }) => {
  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Career Milestones
      </Typography>
      <List>
        {milestones.map((milestone) => (
          <ListItem key={milestone.id}>
            <ListItemText
              primary={milestone.title}
              secondary={
                <Box>
                  <Typography variant="body2">{milestone.description}</Typography>
                  <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                    <Chip label={`Target: ${milestone.targetDate}`} size="small" />
                    <Chip label={`Status: ${milestone.status}`} size="small" />
                    <Chip label={`Priority: ${milestone.priority}`} size="small" />
                  </Box>
                </Box>
              }
            />
            <ListItemSecondaryAction>
              <IconButton edge="end" onClick={() => onUpdateMilestone(milestone)}>
                <Edit />
              </IconButton>
            </ListItemSecondaryAction>
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

interface CareerRoadmapProps {
  milestones: CareerMilestone[];
  onAddMilestone: (milestone: CareerMilestone) => void;
  onUpdateMilestone: (milestone: CareerMilestone) => void;
  onDeleteMilestone: (milestoneId: string) => void;
}

const CareerRoadmap: React.FC<CareerRoadmapProps> = ({ 
  milestones, 
  onAddMilestone, 
  onUpdateMilestone,
  onDeleteMilestone 
}) => {
  // IMPORTANT: Let's force clear the localStorage to reset
  useEffect(() => {
    // This will run only once when the component mounts
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('reset') === 'true') {
      localStorage.removeItem('careerMilestones');
      localStorage.removeItem('careerSkills');
      window.location.href = window.location.pathname; // Refresh without the query param
    }
  }, []);

  const [skills, setSkills] = useState<Skill[]>(() => {
    const savedSkills = localStorage.getItem('careerSkills');
    return savedSkills ? JSON.parse(savedSkills) : [
      { id: '1', name: 'React', currentLevel: 3, targetLevel: 5 },
      { id: '2', name: 'TypeScript', currentLevel: 2, targetLevel: 4 },
      { id: '3', name: 'Node.js', currentLevel: 3, targetLevel: 4 },
    ];
  });

  // Save skills to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('careerSkills', JSON.stringify(skills));
  }, [skills]);

  const [openSkillDialog, setOpenSkillDialog] = useState(false);
  const [openMilestoneDialog, setOpenMilestoneDialog] = useState(false);
  const [editingSkill, setEditingSkill] = useState<Skill | null>(null);
  const [editingMilestone, setEditingMilestone] = useState<CareerMilestone | null>(null);

  // Skill Dialog State
  const [skillName, setSkillName] = useState('');
  const [currentLevel, setCurrentLevel] = useState(1);
  const [targetLevel, setTargetLevel] = useState(3);

  // Milestone Dialog State
  const [milestoneTitle, setMilestoneTitle] = useState('');
  const [milestoneDescription, setMilestoneDescription] = useState('');
  const [milestoneTargetDate, setMilestoneTargetDate] = useState('');
  const [selectedSkillId, setSelectedSkillId] = useState('');
  const [skillLevelIncrease, setSkillLevelIncrease] = useState(1);

  const handleAddSkill = () => {
    setEditingSkill(null);
    setSkillName('');
    setCurrentLevel(1);
    setTargetLevel(3);
    setOpenSkillDialog(true);
  };

  const handleEditSkill = (skill: Skill) => {
    setEditingSkill(skill);
    setSkillName(skill.name);
    setCurrentLevel(skill.currentLevel);
    setTargetLevel(skill.targetLevel);
    setOpenSkillDialog(true);
  };

  const handleDeleteSkill = (id: string) => {
    setSkills(skills.filter(skill => skill.id !== id));
    // We can't directly modify milestones since they come from props
    // Just log that this would require updating the parent component
    console.log('Skill deleted. Associated milestones would need to be removed.');
  };

  const handleSaveSkill = () => {
    if (editingSkill) {
      setSkills(skills.map(skill =>
        skill.id === editingSkill.id
          ? { ...skill, name: skillName, currentLevel, targetLevel }
          : skill
      ));
    } else {
      setSkills([...skills, {
        id: Date.now().toString(),
        name: skillName,
        currentLevel,
        targetLevel,
      }]);
    }
    setOpenSkillDialog(false);
  };

  const handleAddMilestoneLocal = () => {
    if (!milestoneTitle || !selectedSkillId || !milestoneTargetDate) {
      return; // Don't add if required fields are missing
    }
    
    const milestone: CareerMilestone = {
      id: '', // This will be generated by the parent component
      title: milestoneTitle,
      description: milestoneDescription,
      targetDate: milestoneTargetDate,
      status: 'Not Started',
      priority: 'Medium',
      skills: [selectedSkillId],
      resources: [],
      notes: ''
    };
    
    onAddMilestone(milestone);
    setOpenMilestoneDialog(false);
    
    // Reset form
    setMilestoneTitle('');
    setMilestoneDescription('');
    setMilestoneTargetDate('');
    setSelectedSkillId('');
  };

  const handleEditMilestone = (milestone: CareerMilestone) => {
    setEditingMilestone(milestone);
    setMilestoneTitle(milestone.title);
    setMilestoneDescription(milestone.description);
    setMilestoneTargetDate(milestone.targetDate);
    setSelectedSkillId(milestone.skills[0] || '');
    setSkillLevelIncrease(1);
    setOpenMilestoneDialog(true);
  };

  const handleDeleteMilestone = (id: string) => {
    // Call the parent function to delete the milestone
    onDeleteMilestone(id);
  };

  const handleSaveMilestone = () => {
    if (editingMilestone) {
      // Update existing milestone
      const updatedMilestone: CareerMilestone = {
        ...editingMilestone,
        title: milestoneTitle,
        description: milestoneDescription,
        targetDate: milestoneTargetDate,
        skills: [selectedSkillId],
      };
      
      // Call onUpdateMilestone to update the milestone in the parent component
      onUpdateMilestone(updatedMilestone);
    } else {
      // Add new milestone
      handleAddMilestoneLocal();
    }
    
    setOpenMilestoneDialog(false);
    setEditingMilestone(null);
  };

  const handleStatusChange = (milestoneId: string, newStatus: CareerMilestone['status']) => {
    // Find the milestone to update
    const milestone = milestones.find(m => m.id === milestoneId);
    if (!milestone) return;
    
    // Create updated milestone with new status
    const updatedMilestone = {
      ...milestone,
      status: newStatus
    };
    
    // Call onUpdateMilestone to update the milestone in the parent component
    onUpdateMilestone(updatedMilestone);
    
    // Update skills if milestone is now completed
    if (newStatus === 'Completed' && milestone.status !== 'Completed') {
      const skillId = milestone.skills[0] || '';
      const associatedSkill = skills.find(s => s.id === skillId);
      if (associatedSkill) {
        // Use the milestone's skillLevelIncrease if available, otherwise default to 1
        const increase = (milestone as any).skillLevelIncrease || 1;
        
        const newCurrentLevel = Math.min(
          associatedSkill.currentLevel + increase,
          associatedSkill.targetLevel
        );
        
        setSkills(skills.map(skill =>
          skill.id === associatedSkill.id
            ? { ...skill, currentLevel: newCurrentLevel }
            : skill
        ));
      }
    }
  };

  const getSkillName = (skillId: string) => {
    const skill = skills.find(s => s.id === skillId);
    return skill ? skill.name : 'Unknown Skill';
  };

  const renderSkillLevel = (level: number) => {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        {[...Array(5)].map((_, i) => (
          <Star
            key={i}
            sx={{
              color: i < level ? 'gold' : 'grey',
              fontSize: '1.2rem',
            }}
          />
        ))}
      </Box>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Career Roadmap
      </Typography>

      {/* Debug info - remove this in production */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="body2">
          Milestones count: {milestones.length} | 
          <Button 
            size="small" 
            onClick={() => {window.location.href = window.location.pathname + '?reset=true'}}
          >
            Reset All Data
          </Button>
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Skills Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h5">Skills Development Areas</Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={handleAddSkill}
              >
                Add Skill
              </Button>
            </Box>
            <List>
              {skills.map((skill) => (
                <ListItem key={skill.id}>
                  <ListItemText
                    primary={skill.name}
                    secondary={
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2">Current Level:</Typography>
                          {renderSkillLevel(skill.currentLevel)}
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2">Target Level:</Typography>
                          {renderSkillLevel(skill.targetLevel)}
                        </Box>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={() => handleEditSkill(skill)}>
                      <Edit />
                    </IconButton>
                    <IconButton edge="end" onClick={() => handleDeleteSkill(skill.id)}>
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Milestones Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h5">Career Milestones</Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => {
                  setEditingMilestone(null);
                  setMilestoneTitle('');
                  setMilestoneDescription('');
                  setMilestoneTargetDate(new Date().toISOString().split('T')[0]);
                  setSelectedSkillId(skills.length > 0 ? skills[0].id : '');
                  setOpenMilestoneDialog(true);
                }}
              >
                Add Milestone
              </Button>
            </Box>
            <List>
              {milestones.map((milestone) => (
                <ListItem key={milestone.id}>
                  <ListItemText
                    primary={milestone.title}
                    secondary={
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        <Typography variant="body2">{milestone.description}</Typography>
                        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                          <Chip
                            label={`Target: ${new Date(milestone.targetDate).toLocaleDateString()}`}
                            size="small"
                          />
                          <FormControl size="small" sx={{ minWidth: 120 }}>
                            <Select
                              value={milestone.status}
                              onChange={(e) => handleStatusChange(milestone.id, e.target.value as CareerMilestone['status'])}
                              size="small"
                            >
                              <MenuItem value="Not Started">Not Started</MenuItem>
                              <MenuItem value="In Progress">In Progress</MenuItem>
                              <MenuItem value="Completed">Completed</MenuItem>
                            </Select>
                          </FormControl>
                          <Chip
                            label={`Skill: ${getSkillName(milestone.skills[0] || '')}`}
                            size="small"
                            color="primary"
                          />
                        </Box>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={() => handleEditMilestone(milestone)}>
                      <Edit />
                    </IconButton>
                    <IconButton edge="end" onClick={() => handleDeleteMilestone(milestone.id)}>
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
              {milestones.length === 0 && (
                <Box sx={{ p: 2, textAlign: 'center' }}>
                  <Typography color="text.secondary">No milestones yet. Add your first one!</Typography>
                </Box>
              )}
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Skill Dialog */}
      <Dialog open={openSkillDialog} onClose={() => setOpenSkillDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingSkill ? 'Edit Skill' : 'Add Skill'}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                label="Skill Name"
                fullWidth
                value={skillName}
                onChange={(e) => setSkillName(e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <Typography gutterBottom>Current Level</Typography>
              <Slider
                value={currentLevel}
                onChange={(_, value) => setCurrentLevel(value as number)}
                min={1}
                max={5}
                step={1}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            <Grid item xs={12}>
              <Typography gutterBottom>Target Level</Typography>
              <Slider
                value={targetLevel}
                onChange={(_, value) => setTargetLevel(value as number)}
                min={1}
                max={5}
                step={1}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenSkillDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveSkill} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Milestone Dialog */}
      <Dialog open={openMilestoneDialog} onClose={() => setOpenMilestoneDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingMilestone ? 'Edit Milestone' : 'Add Milestone'}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                label="Title"
                fullWidth
                value={milestoneTitle}
                onChange={(e) => setMilestoneTitle(e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Description"
                fullWidth
                multiline
                rows={4}
                value={milestoneDescription}
                onChange={(e) => setMilestoneDescription(e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Target Date"
                type="date"
                fullWidth
                value={milestoneTargetDate}
                onChange={(e) => setMilestoneTargetDate(e.target.value)}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Associated Skill</InputLabel>
                <Select
                  value={selectedSkillId}
                  onChange={(e) => setSelectedSkillId(e.target.value)}
                  label="Associated Skill"
                >
                  {skills.map((skill) => (
                    <MenuItem key={skill.id} value={skill.id}>
                      {skill.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Typography gutterBottom>Skill Level Increase</Typography>
              <Slider
                value={skillLevelIncrease}
                onChange={(_, value) => setSkillLevelIncrease(value as number)}
                min={1}
                max={5}
                step={1}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenMilestoneDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveMilestone} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CareerRoadmap; 