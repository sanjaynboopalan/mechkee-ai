# MechKee.ai - Intelligent AI Assistant

An advanced AI-powered assistant platform that combines intelligent search capabilities with modern conversational AI, featuring a responsive design that works seamlessly across all devices.

## 🚀 Features

- **💬 Intelligent Chat**: Powered by advanced AI models (OpenAI, Groq)
- **🔍 Real-time Search**: Access up-to-date information from the web
- **📄 Document Processing**: Upload and analyze documents with AI
- **🧠 Advanced RAG**: Retrieval-Augmented Generation for accurate responses
- **📱 Mobile Responsive**: Optimized for mobile, tablet, and desktop
- **⚡ Live Data**: Real-time web scraping and content analysis
- **🎯 Personalized Experience**: Adaptive responses based on user preferences

## 🛠 Technology Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for responsive styling
- **Mobile-First Design** with touch-friendly interface
- **GitHub Pages** deployment ready

### Backend
- **FastAPI** - High-performance async Python web framework
- **Python 3.13** with enhanced performance
- **Vector Database** for advanced similarity search
- **AI Models** - OpenAI GPT and Groq integration
- **Docker** containerization

## 📱 Mobile Features

- **Responsive Layout**: Adapts to any screen size (320px - 4K+)
- **Touch Gestures**: Swipe-friendly sidebar navigation
- **Mobile Sidebar**: Collapsible navigation for small screens
- **Optimized Input**: Mobile keyboard-friendly text areas
- **Fast Loading**: Optimized for mobile networks

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/bluemech.git
cd bluemech
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm start
```

### 3. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python scripts/setup.py
uvicorn app.main:app --reload --port 8001
```

### 4. Environment Variables
```bash
cp .env.example .env
# Add your API keys to .env
```

## 🌐 Deployment

### GitHub Pages (Recommended)
```bash
cd frontend
npm run deploy
```

### Manual Deployment
```bash
npm run build
# Upload build/ folder to your hosting service
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/chat` | POST | AI chat interface |
| `/api/search` | POST | Intelligent search |
| `/api/documents` | POST | Document processing |

## 🎨 UI Components

- **ModernChatInterface**: Main chat component with mobile support
- **SearchInterface**: Advanced search with filters
- **DocumentUploader**: Drag-and-drop file upload
- **Responsive Sidebar**: Mobile-friendly navigation

## 📱 Responsive Breakpoints

- **Mobile**: 320px - 768px
- **Tablet**: 768px - 1024px  
- **Desktop**: 1024px+
- **Large Desktop**: 1440px+

## 🔧 Configuration

### API Keys (.env)
```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
SERPER_API_KEY=your_serper_key
```

### Build Settings
```json
{
  "homepage": "https://yourusername.github.io/bluemech",
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d build"
  }
}
```

## 📁 Project Structure

```
bluemech/
├── frontend/                 # React TypeScript app
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ModernChatInterface.tsx
│   │   │   ├── SearchInterface.tsx
│   │   │   └── DocumentUploader.tsx
│   │   ├── services/        # API services
│   │   └── styles/          # CSS styles
│   ├── build/               # Production build
│   └── package.json
├── backend/                 # FastAPI backend
│   ├── app/                # Application code
│   ├── core/               # AI functionality
│   └── requirements.txt
└── README.md
```

## 🚀 Performance

- **Build Size**: ~120KB (optimized)
- **Load Time**: <2s on 3G
- **Mobile Score**: 95+ (Lighthouse)
- **Desktop Score**: 98+ (Lighthouse)

## 🛠 Development

### Local Development
```bash
# Frontend (port 3000)
cd frontend && npm start

# Backend (port 8001)  
cd backend && uvicorn app.main:app --reload --port 8001
```

### Build for Production
```bash
cd frontend
npm run build
```

## 📋 Roadmap

- [ ] PWA Support
- [ ] Offline Mode
- [ ] Voice Chat
- [ ] Multi-language Support
- [ ] Advanced Analytics
- [ ] API Rate Limiting

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@mechkee.ai
- 💬 GitHub Issues: [Report Bug](https://github.com/yourusername/bluemech/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/bluemech/wiki)

## 🌟 Acknowledgments

- OpenAI for GPT models
- Groq for fast AI inference
- React team for the amazing framework
- Tailwind CSS for utility-first styling

---

**MechKee.ai** - Where Intelligence Meets Simplicity 🤖✨

![Mobile Responsive](https://img.shields.io/badge/Mobile-Responsive-green)
![React](https://img.shields.io/badge/React-18-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Ready-success)