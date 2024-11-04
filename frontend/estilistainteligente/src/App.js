import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [numRecommendations, setNumRecommendations] = useState(1);
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: 'Bem-vindo! Por favor, envie uma imagem para receber recomendações.',
    },
  ]);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImage(file);
    setMessages((prevMessages) => [
      ...prevMessages,
      { type: 'user', text: 'Imagem enviada.' },
      { type: 'bot', text: 'Quantas recomendações você deseja?' },
    ]);
  };

  const handleNumRecommendationsChange = (event) => {
    setNumRecommendations(event.target.value);
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('numRecommendations', numRecommendations);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      setRecommendations(data.recommendations);
      setMessages((prevMessages) => [
        ...prevMessages,
        { type: 'user', text: `Recomendações: ${numRecommendations}` },
        { type: 'bot', text: 'Aqui estão suas recomendações:' },
      ]);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    }
  };

  return (
    <div className="app-container">
      <div className="chatbot-container">
        <div className="chatbot-header">
          <h2>Estilista Inteligente</h2>
        </div>
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.type}`}>
              <p>{msg.text}</p>
            </div>
          ))}
          {image && (
            <div className="chat-message user">
              <img src={URL.createObjectURL(image)} alt="Uploaded preview" className="uploaded-image" />
            </div>
          )}
        </div>
        <div className="chat-input">
          {!image ? (
            <input type="file" accept="image/*" onChange={handleImageUpload} className="file-input" />
          ) : (
            <div className="input-group">
              <input
                type="number"
                min="1"
                max="10"
                value={numRecommendations}
                onChange={handleNumRecommendationsChange}
                className="number-input"
              />
              <button onClick={handleSubmit} className="submit-button">
                Obter Recomendações
              </button>
            </div>
          )}
        </div>
        {recommendations.length > 0 && (
          <div className="recommendation-gallery">
            {recommendations.map((item, index) => (
              <div key={index} className="recommendation-item">
                <img src={item} alt={`Recommendation ${index}`} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
