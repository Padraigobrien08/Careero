import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  Grid,
} from '@mui/material';
import {
  Add,
  Edit,
  Delete,
  CalendarToday,
  Business,
  LocationOn,
} from '@mui/icons-material';

interface JobApplication {
  id: string;
  company: string;
  position: string;
  location: string;
  status: 'Applied' | 'Interview' | 'Offer' | 'Rejected';
  appliedDate: string;
  notes: string;
}

const JobApplications: React.FC = () => {
  const [applications, setApplications] = useState<JobApplication[]>([
    {
      id: '1',
      company: 'Tech Corp',
      position: 'Senior Frontend Developer',
      location: 'San Francisco, CA',
      status: 'Interview',
      appliedDate: '2024-03-15',
      notes: 'Technical interview scheduled for next week',
    },
    {
      id: '2',
      company: 'Startup Inc',
      position: 'Full Stack Developer',
      location: 'Remote',
      status: 'Applied',
      appliedDate: '2024-03-20',
      notes: 'Waiting for response',
    },
  ]);

  const [openDialog, setOpenDialog] = useState(false);
  const [editingApplication, setEditingApplication] = useState<JobApplication | null>(null);

  const statusColors = {
    Applied: 'primary',
    Interview: 'warning',
    Offer: 'success',
    Rejected: 'error',
  };

  const handleAddApplication = () => {
    setEditingApplication(null);
    setOpenDialog(true);
  };

  const handleEditApplication = (application: JobApplication) => {
    setEditingApplication(application);
    setOpenDialog(true);
  };

  const handleDeleteApplication = (id: string) => {
    setApplications(applications.filter(app => app.id !== id));
  };

  const handleSaveApplication = (application: Omit<JobApplication, 'id'>) => {
    if (editingApplication) {
      setApplications(applications.map(app =>
        app.id === editingApplication.id ? { ...application, id: app.id } : app
      ));
    } else {
      setApplications([...applications, { ...application, id: Date.now().toString() }]);
    }
    setOpenDialog(false);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Job Applications Tracker
      </Typography>

      <Box sx={{ mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={handleAddApplication}
        >
          Add Application
        </Button>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Company</TableCell>
              <TableCell>Position</TableCell>
              <TableCell>Location</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Applied Date</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {applications.map((application) => (
              <TableRow key={application.id}>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Business sx={{ mr: 1 }} />
                    {application.company}
                  </Box>
                </TableCell>
                <TableCell>{application.position}</TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <LocationOn sx={{ mr: 1 }} />
                    {application.location}
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip
                    label={application.status}
                    color={statusColors[application.status] as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <CalendarToday sx={{ mr: 1 }} />
                    {new Date(application.appliedDate).toLocaleDateString()}
                  </Box>
                </TableCell>
                <TableCell>
                  <IconButton
                    size="small"
                    onClick={() => handleEditApplication(application)}
                  >
                    <Edit />
                  </IconButton>
                  <IconButton
                    size="small"
                    onClick={() => handleDeleteApplication(application.id)}
                  >
                    <Delete />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <ApplicationDialog
        open={openDialog}
        onClose={() => setOpenDialog(false)}
        onSave={handleSaveApplication}
        application={editingApplication}
      />
    </Box>
  );
};

interface ApplicationDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (application: Omit<JobApplication, 'id'>) => void;
  application: JobApplication | null;
}

const ApplicationDialog = ({ open, onClose, onSave, application }: ApplicationDialogProps) => {
  const [company, setCompany] = useState(application?.company || '');
  const [position, setPosition] = useState(application?.position || '');
  const [location, setLocation] = useState(application?.location || '');
  const [status, setStatus] = useState(application?.status || 'Applied');
  const [appliedDate, setAppliedDate] = useState(application?.appliedDate || '');
  const [notes, setNotes] = useState(application?.notes || '');

  const handleSave = () => {
    onSave({
      company,
      position,
      location,
      status,
      appliedDate,
      notes,
    });
    setCompany('');
    setPosition('');
    setLocation('');
    setStatus('Applied');
    setAppliedDate('');
    setNotes('');
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{application ? 'Edit Application' : 'Add Application'}</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12}>
            <TextField
              label="Company"
              fullWidth
              value={company}
              onChange={(e) => setCompany(e.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              label="Position"
              fullWidth
              value={position}
              onChange={(e) => setPosition(e.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              label="Location"
              fullWidth
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              select
              label="Status"
              fullWidth
              value={status}
              onChange={(e) => setStatus(e.target.value as JobApplication['status'])}
            >
              <MenuItem value="Applied">Applied</MenuItem>
              <MenuItem value="Interview">Interview</MenuItem>
              <MenuItem value="Offer">Offer</MenuItem>
              <MenuItem value="Rejected">Rejected</MenuItem>
            </TextField>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              label="Applied Date"
              type="date"
              fullWidth
              value={appliedDate}
              onChange={(e) => setAppliedDate(e.target.value)}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              label="Notes"
              fullWidth
              multiline
              rows={4}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained">
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default JobApplications; 