// frontend/src/App.jsx
import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      // Connect to Flask Backend
      const response = await axios.post('http://localhost:5000/predict', {
        text: inputText
      });
      setResult(response.data.prediction);
    } catch (err) {
      console.error(err);
      setError('Failed to connect to the server or analyze text.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>AI Fake News Detector</h1>
        <p>Using Fine-Tuned BERT Transformer</p>
      </header>

      <main className="main-content">
        <textarea
          className="text-input"
          rows="6"
          placeholder="Paste the news article content here..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />

        <button 
          className="analyze-btn" 
          onClick={handleAnalyze} 
          disabled={loading || !inputText}
        >
          {loading ? 'Analyzing...' : 'Detect Veracity'}
        </button>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <div className={`result-box ${result.label.toLowerCase()}`}>
            <h2>Prediction: {result.label}</h2>
            <div className="confidence-bar">
              <div 
                className="fill" 
                style={{ width: `${result.confidence}%` }}
              ></div>
            </div>
            <p>Confidence: <strong>{result.confidence}%</strong></p>
            <details>
              <summary>View Technical Details</summary>
              <p>Raw Fake Probability: {result.raw_scores.fake_prob}</p>
              <p>Raw Real Probability: {result.raw_scores.real_prob}</p>
            </details>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;