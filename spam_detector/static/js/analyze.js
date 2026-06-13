// analyze.js — SpamShield email analyzer

const SAMPLES = {
  spam: {
    subject: "CONGRATULATIONS! You've Won $1,000,000 — Claim Now!!!",
    body: `Dear Lucky Winner,

You have been RANDOMLY SELECTED as the winner of our International Lottery Programme!
Your prize is worth ONE MILLION DOLLARS (USD $1,000,000).

To claim your prize IMMEDIATELY, click the link below and submit your personal details including full name, bank account number, and date of birth.

This offer EXPIRES IN 24 HOURS — do not delay!!! Act NOW!!!

CLICK HERE TO CLAIM: http://totally-legit-prize-claim.example.com

Yours sincerely,
International Lottery Foundation`
  },
  ham: {
    subject: "Re: Team meeting notes — follow-up on action items",
    body: `Hi everyone,

Thanks for joining the standup this morning. Here are the key action items we agreed on:

1. Alice will finish the API documentation by Thursday.
2. Bob will set up the staging environment before the end of the week.
3. Carol will coordinate with the design team on the new dashboard mockups.

Our next sync is scheduled for Friday at 2 PM. Please review your items beforehand and flag any blockers in the group chat.

Best,
David`
  }
};

function loadSample(type) {
  const s = SAMPLES[type];
  document.getElementById('emailSubject').value = s.subject;
  document.getElementById('emailBody').value    = s.body;
  updateCharCount();
  clearResult();
}

function updateCharCount() {
  const body = document.getElementById('emailBody').value;
  document.getElementById('charCount').textContent = body.length.toLocaleString();
}

function clearForm() {
  document.getElementById('emailSubject').value = '';
  document.getElementById('emailBody').value    = '';
  document.getElementById('charCount').textContent = '0';
  clearResult();
  document.getElementById('formError').classList.add('hidden');
}

function clearResult() {
  document.getElementById('resultContent').classList.add('hidden');
  document.getElementById('resultPlaceholder').classList.remove('hidden');
}

function setLoading(loading) {
  const btn     = document.getElementById('analyzeBtn');
  const text    = document.getElementById('btnText');
  const spinner = document.getElementById('btnSpinner');
  btn.disabled  = loading;
  text.textContent = loading ? 'Analyzing…' : '🔍 Check for Spam';
  spinner.classList.toggle('hidden', !loading);
}

async function analyzeEmail() {
  const subject = document.getElementById('emailSubject').value.trim();
  const body    = document.getElementById('emailBody').value.trim();
  const errBox  = document.getElementById('formError');

  if (!body) {
    errBox.classList.remove('hidden');
    document.getElementById('emailBody').focus();
    return;
  }
  errBox.classList.add('hidden');
  setLoading(true);

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ subject, body })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    showResult(data);
  } catch (err) {
    errBox.textContent = '⚠ ' + (err.message || 'An error occurred. Please try again.');
    errBox.classList.remove('hidden');
  } finally {
    setLoading(false);
  }
}

function showResult(data) {
  // Hide placeholder, show content
  document.getElementById('resultPlaceholder').classList.add('hidden');
  document.getElementById('resultContent').classList.remove('hidden');

  // Verdict banner
  const banner = document.getElementById('verdictBanner');
  banner.className = 'verdict-banner ' + (data.is_spam ? 'spam-result' : 'ham-result');
  document.getElementById('verdictIcon').textContent  = data.is_spam ? '🚨' : '✅';
  document.getElementById('verdictLabel').textContent = data.label;

  // Confidence
  document.getElementById('confValue').textContent = data.spam_probability + '%';
  const bar = document.getElementById('gaugeBar');
  const pct = data.spam_probability;
  // Colour: green → orange → red
  const barColor = pct > 70 ? '#ff4c4c' : pct > 40 ? '#f0883e' : '#39d353';
  bar.style.width      = pct + '%';
  bar.style.background = barColor;

  // Confidence badge
  const badge = document.getElementById('confLabel');
  badge.textContent = data.confidence_label;
  badge.style.background = (pct > 70 ? 'rgba(255,76,76,.15)' : pct > 40 ? 'rgba(240,136,62,.15)' : 'rgba(57,211,83,.15)');
  badge.style.color  = barColor;
  badge.style.border = '1px solid ' + barColor;
  badge.style.padding = '2px 10px';
  badge.style.borderRadius = '10px';

  // Probabilities
  document.getElementById('hamProb').textContent  = data.ham_probability  + '%';
  document.getElementById('spamProb').textContent = data.spam_probability + '%';

  // Tip
  document.getElementById('tipText').textContent = data.tip;
}

// Char counter listener
document.getElementById('emailBody').addEventListener('input', updateCharCount);

// Allow Ctrl+Enter to submit
document.getElementById('emailBody').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) analyzeEmail();
});
