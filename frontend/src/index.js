import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Get the root element from the HTML file
const rootElement = document.getElementById('root');

// Create a React root and render the App component
const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);