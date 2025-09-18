import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, Plus } from 'lucide-react';
import Button from './ui/Button';
import Card from './ui/Card';
import { enhancedAPI } from '../services/enhancedapi';
import toast from 'react-hot-toast';

interface DocumentUploaderProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: () => void;
}

const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  isOpen,
  onClose,
  onUploadSuccess
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [url, setUrl] = useState('');
  const [isAddingUrl, setIsAddingUrl] = useState(false);

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const file = files[0];
    const allowedTypes = ['.txt', '.pdf', '.docx', '.md', '.html'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();

    if (!allowedTypes.includes(fileExtension)) {
      toast.error(`Unsupported file type. Allowed: ${allowedTypes.join(', ')}`);
      return;
    }

    setIsUploading(true);
    try {
      const response = await enhancedAPI.uploadDocument(file, file.name);
      toast.success(`Document "${file.name}" uploaded successfully! Created ${response.chunks_created} chunks.`);
      onUploadSuccess();
      onClose();
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload document. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleUrlAdd = async () => {
    if (!url.trim()) return;

    setIsAddingUrl(true);
    try {
      await enhancedAPI.addURL(url, url);
      toast.success(`URL "${url}" added to knowledge base successfully!`);
      setUrl('');
      onUploadSuccess();
      onClose();
    } catch (error) {
      console.error('Add URL error:', error);
      toast.error('Failed to add URL. Please check the URL and try again.');
    } finally {
      setIsAddingUrl(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFileUpload(e.dataTransfer.files);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.2 }}
          className="w-full max-w-md"
        >
          <Card className="relative p-6">
            <Button
              onClick={onClose}
              className="absolute top-4 right-4 p-1 text-gray-400 hover:text-white"
              variant="ghost"
              size="sm"
            >
              <X className="w-4 h-4" />
            </Button>

            <h3 className="text-lg font-semibold mb-4 gradient-text">Add to Knowledge Base</h3>

            {/* File Upload Section */}
            <div className="mb-6">
              <h4 className="text-sm font-medium mb-3 text-gray-300">Upload Document</h4>
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-lg p-6 text-center transition-all ${
                  isDragOver
                    ? 'border-blue-400 bg-blue-400/10'
                    : 'border-gray-600 hover:border-gray-500'
                }`}
              >
                <Upload className="w-8 h-8 mx-auto mb-3 text-gray-400" />
                <p className="text-sm text-gray-400 mb-3">
                  Drag & drop files here, or{' '}
                  <label className="text-blue-400 cursor-pointer hover:text-blue-300">
                    browse
                    <input
                      type="file"
                      className="hidden"
                      accept=".txt,.pdf,.docx,.md,.html"
                      onChange={(e) => handleFileUpload(e.target.files)}
                      disabled={isUploading}
                    />
                  </label>
                </p>
                <p className="text-xs text-gray-500">
                  Supported: TXT, PDF, DOCX, MD, HTML
                </p>
              </div>
            </div>

            {/* URL Addition Section */}
            <div className="mb-6">
              <h4 className="text-sm font-medium mb-3 text-gray-300">Add Web Page</h4>
              <div className="flex space-x-2">
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/article"
                  className="flex-1 input-modern text-sm"
                  disabled={isAddingUrl}
                />
                <Button
                  onClick={handleUrlAdd}
                  disabled={!url.trim() || isAddingUrl}
                  className="btn-primary px-3"
                >
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Status */}
            {(isUploading || isAddingUrl) && (
              <div className="flex items-center justify-center space-x-2 text-sm text-gray-400">
                <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                <span>{isUploading ? 'Uploading document...' : 'Adding URL...'}</span>
              </div>
            )}
          </Card>
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default DocumentUploader;