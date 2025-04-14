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
  CircularProgress,
  Alert,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tabs,
  Tab,
  Divider,
} from '@mui/material';
import { Add, Delete, Description, Save, Visibility } from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

interface Resume {
  id: string;
  name: string;
  path: string;
  uploadedAt: string;
}

interface ParsedResume {
  text: string;
  skills: string[];
  experience: any[];
  education: any[];
}

const ResumeEditor: React.FC = () => {
  const [resumes, setResumes] = useState<Resume[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openUploadDialog, setOpenUploadDialog] = useState(false);
  const [previewFile, setPreviewFile] = useState<File | null>(null);
  const [openPreviewDialog, setOpenPreviewDialog] = useState(false);
  const [currentResume, setCurrentResume] = useState<Resume | null>(null);
  const [parsedResume, setParsedResume] = useState<ParsedResume | null>(null);
  const [previewTab, setPreviewTab] = useState(0);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setPreviewFile(acceptedFiles[0]);
      }
    },
  });

  useEffect(() => {
    fetchResumes();
  }, []);

  const fetchResumes = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch('http://localhost:8000/resumes');
      if (!response.ok) {
        throw new Error('Failed to fetch resumes');
      }
      const data = await response.json();
      setResumes(data.resumes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadResume = async () => {
    if (!previewFile) return;

    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', previewFile);
      
      // Add a dummy token for authorization
      const token = localStorage.getItem('authToken') || 'dummy-token';

      const response = await fetch('http://localhost:8000/upload-resume', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload resume');
      }

      const data = await response.json();
      console.log('Upload response:', data);  // Add logging to debug
      
      await fetchResumes();
      setOpenUploadDialog(false);
      setPreviewFile(null);
    } catch (err) {
      console.error('Upload error:', err);  // Add error logging
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteResume = async (id: string) => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`http://localhost:8000/resumes/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete resume');
      }

      await fetchResumes();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch (e) {
      return 'Unknown date';
    }
  };

  const getMostRecentResume = () => {
    return resumes.length > 0 ? resumes[0] : null;
  };

  const handleViewResume = async (resume: Resume) => {
    try {
      setLoading(true);
      setError(null);
      setCurrentResume(resume);
      
      const response = await fetch(`http://localhost:8000/resumes/${resume.id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch resume');
      }
      
      const data = await response.json();
      setParsedResume(data.parsed_content);
      setOpenPreviewDialog(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to view resume');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadResume = async (resume: Resume) => {
    try {
      const response = await fetch(`http://localhost:8000/resumes/${resume.id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch resume');
      }
      const data = await response.json();
      
      // Convert base64 to blob
      const byteCharacters = atob(data.file.content);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: data.file.content_type });
      
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = data.file.filename;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download resume');
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Resume Editor
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          {/* Most Recent Resume Section */}
          {getMostRecentResume() && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Most Recent Resume
              </Typography>
              <List>
                <ListItem>
                  <ListItemText
                    primary={getMostRecentResume()?.name}
                    secondary={`Uploaded: ${formatDate(getMostRecentResume()?.uploadedAt || '')}`}
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={() => handleViewResume(getMostRecentResume()!)}>
                      <Description />
                    </IconButton>
                    <IconButton edge="end" onClick={() => handleDeleteResume(getMostRecentResume()?.id || '')}>
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </Paper>
          )}

          {/* All Resumes Section */}
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">All Resumes</Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => setOpenUploadDialog(true)}
              >
                Add New Resume
              </Button>
            </Box>
            <List>
              {resumes.map((resume) => (
                <ListItem key={resume.id}>
                  <ListItemText
                    primary={resume.name}
                    secondary={`Uploaded: ${formatDate(resume.uploadedAt)}`}
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={() => handleViewResume(resume)}>
                      <Description />
                    </IconButton>
                    <IconButton edge="end" onClick={() => handleDeleteResume(resume.id)}>
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Paper>
        </>
      )}

      {/* Upload Dialog */}
      <Dialog open={openUploadDialog} onClose={() => setOpenUploadDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Upload New Resume</DialogTitle>
        <DialogContent>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 1,
              p: 3,
              textAlign: 'center',
              cursor: 'pointer',
              mb: 2,
            }}
          >
            <input {...getInputProps()} />
            <Description sx={{ fontSize: 48, color: 'grey.400', mb: 1 }} />
            <Typography>
              {isDragActive
                ? 'Drop the resume here'
                : 'Drag and drop a resume file here, or click to select'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Supported formats: PDF, DOC, DOCX
            </Typography>
          </Box>

          {previewFile && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">Preview:</Typography>
              <Typography variant="body2">{previewFile.name}</Typography>
              <Typography variant="body2" color="text.secondary">
                Size: {(previewFile.size / 1024 / 1024).toFixed(2)} MB
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setOpenUploadDialog(false);
            setPreviewFile(null);
          }}>
            Cancel
          </Button>
          <Button
            onClick={handleUploadResume}
            variant="contained"
            disabled={!previewFile || loading}
            startIcon={<Save />}
          >
            Save Resume
          </Button>
        </DialogActions>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog 
        open={openPreviewDialog} 
        onClose={() => setOpenPreviewDialog(false)} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle>
          {currentResume?.name}
          <Typography variant="subtitle2" color="text.secondary">
            Uploaded: {currentResume ? formatDate(currentResume.uploadedAt) : ''}
          </Typography>
        </DialogTitle>
        <DialogContent>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : parsedResume ? (
            <Box>
              <Tabs value={previewTab} onChange={(_, newValue) => setPreviewTab(newValue)}>
                <Tab label="Text" />
                <Tab label="Skills" />
                <Tab label="Experience" />
                <Tab label="Education" />
              </Tabs>
              
              <Box sx={{ mt: 2 }}>
                {previewTab === 0 && (
                  <Paper sx={{ p: 2, whiteSpace: 'pre-wrap' }}>
                    {parsedResume.text}
                  </Paper>
                )}
                
                {previewTab === 1 && (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {parsedResume.skills.map((skill, index) => (
                      <Paper key={index} sx={{ p: 1, backgroundColor: 'primary.light', color: 'primary.contrastText' }}>
                        {skill}
                      </Paper>
                    ))}
                  </Box>
                )}
                
                {previewTab === 2 && (
                  <List>
                    {parsedResume.experience.map((exp, index) => (
                      <ListItem key={index} sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                        <Typography variant="subtitle1" fontWeight="bold">
                          {exp.title} at {exp.company}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {exp.duration}
                        </Typography>
                        <Typography variant="body1">
                          {exp.description}
                        </Typography>
                      </ListItem>
                    ))}
                  </List>
                )}
                
                {previewTab === 3 && (
                  <List>
                    {parsedResume.education.map((edu, index) => (
                      <ListItem key={index} sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                        <Typography variant="subtitle1" fontWeight="bold">
                          {edu.degree}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {edu.institution} • {edu.year}
                        </Typography>
                      </ListItem>
                    ))}
                  </List>
                )}
              </Box>
            </Box>
          ) : (
            <Typography>No resume content available</Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenPreviewDialog(false)}>Close</Button>
          <Button 
            onClick={() => currentResume && handleDownloadResume(currentResume)}
            variant="contained"
          >
            Download
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ResumeEditor; 