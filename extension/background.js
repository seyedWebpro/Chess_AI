// background.js: Manifest V3 Service Worker for Chess AI Extension
chrome.runtime.onInstalled.addListener(() => {
  console.log('[ChessAI Service Worker] Extension installed and active.');
  chrome.storage.local.set({
    persona: '1500',
    voiceMode: 'EARCON',
    overlay: true,
    autoMouse: false
  });
});
