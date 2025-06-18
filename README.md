# AI Gift Recommendation System

An intelligent gift recommendation system that analyzes community discussions to provide personalized product suggestions with direct purchase links.

## 🌟 Features

- **AI-Powered Analysis**: Uses GPT-4 to understand gift requirements
- **Community Intelligence**: Analyzes Reddit discussions for real recommendations
- **Smart Product Search**: Finds actual products from Indian e-commerce sites
- **Interactive Chat Interface**: Easy-to-use conversational interface
- **Secure Deployment**: Environment-based API key management

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gift-recommendation-system.git
   cd gift-recommendation-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Visit** `http://localhost:5000`

## 🔑 API Keys Setup

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login → API Keys → Create new secret key
3. Add to `.env`: `OPENAI_API_KEY=sk-...`

### Google Custom Search API
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Custom Search API
3. Create credentials (API Key)
4. Set up Custom Search Engine at [CSE](https://cse.google.com/)
5. Add to `.env`: `GOOGLE_API_KEY=...` and `SEARCH_ENGINE_ID=...`

### Reddit API
1. Visit [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Create new application (script type)
3. Note client ID and secret
4. Add to `.env`: `REDDIT_CLIENT_ID=...` and `REDDIT_CLIENT_SECRET=...`

## 🌐 Deployment on Render

### 1. Prepare for GitHub
```bash
# Make sure .env is in .gitignore
echo ".env" >> .gitignore

# Commit and push
git add .
git commit -m "Initial commit with secure API key management"
git push origin main
```

### 2. Deploy on Render
1. Go to [Render.com](https://render.com) and connect GitHub
2. Create new **Web Service**
3. Select your repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment**: `production`

### 3. Set Environment Variables in Render
In your Render dashboard → Environment, add:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `SEARCH_ENGINE_ID`
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `SECRET_KEY`
- `FLASK_ENV=production`

## 🛡️ Security Features

✅ **Environment Variables**: All API keys stored securely  
✅ **Git Ignore**: Sensitive files excluded from version control  
✅ **Error Handling**: Graceful handling of missing credentials  
✅ **Input Validation**: Secure processing of user inputs  
✅ **Rate Limiting**: Built-in delays to respect API limits  

## 📁 Project Structure

```
gift-recommendation-system/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Files to exclude from Git
├── README.md             # Project documentation
└── templates/
    └── index.html        # Frontend interface
```

## 🔧 Configuration

The system supports different environments:

- **Development**: Debug mode enabled, detailed logging
- **Production**: Optimized for deployment, error handling

Set `FLASK_ENV=development` for local development or `FLASK_ENV=production` for deployment.

## 🔍 Testing APIs

Visit `/api/test_apis` to check if all your API keys are working correctly.

## 💡 How It Works

1. **User Input**: User describes gift requirements
2. **Clarification**: AI asks follow-up questions
3. **Reddit Analysis**: Searches and analyzes community discussions
4. **AI Processing**: GPT-4 extracts product recommendations
5. **Product Search**: Finds actual products on e-commerce sites
6. **Results**: Displays products with direct purchase links

## 🚨 Important Security Notes

- **Never commit API keys** to version control
- **Use environment variables** for all sensitive data
- **Regularly rotate** your API keys
- **Monitor API usage** and costs
- **Keep dependencies updated** for security patches

## 📝 Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | Yes |
| `GOOGLE_API_KEY` | Google Custom Search API key | Yes |
| `SEARCH_ENGINE_ID` | Google Custom Search Engine ID | Yes |
| `REDDIT_CLIENT_ID` | Reddit application client ID | Yes |
| `REDDIT_CLIENT_SECRET` | Reddit application secret | Yes |
| `REDDIT_USER_AGENT` | Reddit API user agent string | Yes |
| `SECRET_KEY` | Flask session secret key | Yes |
| `FLASK_ENV` | Environment (development/production) | No |
| `PORT` | Server port (default: 5000) | No |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 💬 Support

If you encounter issues:
1. Check your API keys are correctly set
2. Verify all environment variables
3. Check the logs for specific error messages
4. Ensure you have sufficient API credits

---