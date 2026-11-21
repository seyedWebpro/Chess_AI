// popup.js: Extension configuration and state management
document.addEventListener('DOMContentLoaded', () => {
  const personaSelect = document.getElementById('persona-select');
  const voiceSelect = document.getElementById('voice-select');
  const toggleOverlay = document.getElementById('toggle-overlay');
  const toggleAutoMouse = document.getElementById('toggle-automouse');
  const statusBadge = document.getElementById('ext-status');

  // Load saved preferences from chrome.storage
  if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(['persona', 'voiceMode', 'overlay', 'autoMouse'], (res) => {
      if (res.persona) personaSelect.value = res.persona;
      if (res.voiceMode) voiceSelect.value = res.voiceMode;
      if (res.overlay !== undefined) toggleOverlay.checked = res.overlay;
      if (res.autoMouse !== undefined) toggleAutoMouse.checked = res.autoMouse;
    });
  }

  // Save changes
  function saveSettings() {
    const settings = {
      persona: personaSelect.value,
      voiceMode: voiceSelect.value,
      overlay: toggleOverlay.checked,
      autoMouse: toggleAutoMouse.checked
    };
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
      chrome.storage.local.set(settings);
    }
  }

  personaSelect.addEventListener('change', saveSettings);
  voiceSelect.addEventListener('change', saveSettings);
  toggleOverlay.addEventListener('change', saveSettings);
  toggleAutoMouse.addEventListener('change', saveSettings);
});
