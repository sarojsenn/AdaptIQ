# AdaptIQ

**Hack-A-Thon: AI For Education - AI In Education**  
Organized by Unstop

**Team:** Bruteforce Army 

**Contributors:** 
- Saroj Sen
- Tiyasa Saha
- Soumava Das

---

## 🚀 Overview

AdaptIQ is an AI-powered adaptive assessment platform designed exclusively for students. It leverages advanced psychometric and machine learning models to deliver personalized, real-time assessments and actionable insights through an interactive student dashboard.

- **Note:**  
	- The current model is trained on a basic dataset for demonstration. It can be retrained with more advanced datasets in the future.
	- This platform is only for students. There are no teacher features or dashboards.

---

## ✨ Features

- **Adaptive Assessment:**  
	Questions adapt in real-time to each student's ability, maximizing learning and engagement.

- **IRT-Based Analysis:**  
	Uses a 2-Parameter Logistic (2PL) Item Response Theory model for precise ability estimation and question selection.

- **Bayesian Knowledge Tracing:**  
	Tracks student knowledge over time, identifying strengths and gaps.

- **Smart Question Selection:**  
	Selects questions with the highest information gain, ensuring efficient and effective assessment.

- **Real-Time Analytics:**  
	Interactive student dashboard displays progress charts, accuracy trends, skill mastery, and personalized recommendations.

- **Lightning Fast API:**  
	RESTful backend with CORS support for seamless integration.

- **Secure & Scalable:**  
	Enterprise-grade security and scalable architecture.

- **Modern UI:**  
	Beautiful, responsive frontend built with Tailwind CSS and interactive navigation.

---

## 🏗️ Tech Stack

- **Frontend:**  
	- HTML, Tailwind CSS, JavaScript  
	- Responsive, glass-effect UI  
	- Interactive navigation and smooth scrolling

- **Backend:**  
	- Python (Flask/Express)  
	- Machine learning models: IRT, BKT  
	- RESTful API

- **Data:**  
	- Trained on a basic sample dataset (can be extended)

---

## 🧑‍🎓 Student Dashboard

- Personalized dashboard for each student
- Visualizes progress, accuracy, and skill mastery

---

## ⚡ Quick Start

# AdaptIQ

Adaptive assessment platform (student-facing) — demo implementation combining adaptive question selection, simple IRT/BKT-style modeling and a Node + Python backend with a static Tailwind frontend.

This repository contains two backend approaches used in the project:
- `server.js` — primary Node/Express backend with user registration, OTP email verification, and MongoDB integration.
- `api_server.py` and `adaptive_api_server.py` — lightweight Flask demo APIs used for quick local testing and demoing the adaptive question selection logic.

Status: demo / proof-of-concept. See `data/` for question data and `client/pages/` for the static frontend pages.

---

## Quick facts

- Project type: demo adaptive-assessment (student-only)
- Frontend: static HTML + Tailwind (located in `client/pages`)
- Backends: Node/Express (`server.js`) and Flask (`api_server.py` / `adaptive_api_server.py`)
- Data: CSV datasets under `data/` (e.g. `data/questions.csv`)
- Tests: a few Python & JS test files are present for local verification

---

## Requirements

- Node.js (>= 16) and npm
- Python 3.8+ (for the Flask demo servers)
- MongoDB (optional — required only when running the Node server with DB features)

Environment variables used by `server.js` (put in a `.env` file in the project root):

- `MONGODB_URI` — MongoDB connection string (optional for demo/static flows)
- `EMAIL_USER` and `EMAIL_PASS` — Gmail credentials used for OTP email sending
- `JWT_SECRET` — JSON Web Token secret (required and recommended length >= 32 characters)

Do not commit real secrets to the repository.

---

## Setup & run (PowerShell)

Below are quick steps to run either the Node backend (recommended for full functionality) or the Python demo server.

1) Install Node dependencies and run the Node server

```powershell
cd C:\Users\KIIT\Desktop\AdaptIQ
npm install
# create a .env with MONGODB_URI, EMAIL_USER, EMAIL_PASS, JWT_SECRET
npm start
```

The Node server will serve the static frontend from `client/` and provide the full registration/OTP/login endpoints.

2) Run the Flask demo API (useful for quick local testing without MongoDB/email config)

```powershell
cd C:\Users\KIIT\Desktop\AdaptIQ
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If you have a requirements.txt, install it. If not, install the essentials:
pip install flask flask-cors pandas numpy joblib
# Start demo server (one of the demo endpoints):
python api_server.py
# or
python adaptive_api_server.py
```

After either backend is running, open a browser at the local address printed in the terminal and/or open the static pages in `client/pages/` (e.g. `LandingPage.html`).

---

## Project layout (important files)

- `server.js` — Node/Express backend (full user flows, OTP, MongoDB)
- `package.json` — Node dependency manifest and start scripts
- `api_server.py` — Flask demo server with a small in-memory demo model and endpoints (`/api/start-assessment`, `/api/get-question`, `/api/submit-response`, `/api/health`)
- `adaptive_api_server.py` — Flask demo that loads `data/questions.csv`, generates missing options and demonstrates adaptive selection logic
- `client/` — static frontend (Tailwind CSS, sample pages)
- `data/` — question CSVs and datasets used by the demo
- `test_*` files — local tests & examples (Python and JS)

---

## Tests and quick checks

- There are a number of test files in the repo (for example `test_questions_with_options.py`, `test_option_generation.py`, and some JS test pages). They are mostly small local scripts used during development.

To run Python tests (if you use pytest):

```powershell
cd C:\Users\KIIT\Desktop\AdaptIQ
# optional: Activate your virtualenv
pip install pytest
pytest -q
```

Node tests / dev scripts (manual):

```powershell
npm run dev  # requires nodemon (devDependency)
```

---

## Notes & troubleshooting

- If `server.js` fails at startup with a JWT_SECRET error, add a `JWT_SECRET` with at least 32 characters to your `.env`.
- If email sending fails, check `EMAIL_USER`/`EMAIL_PASS` and consider using an app-specific password for Gmail.
- `adaptive_api_server.py` includes robust CSV-reading logic (tries several encodings). If `data/questions.csv` doesn't load, inspect the CSV headers and the `id` column.

---

## Contributing

Small, focused PRs welcome. Suggested follow-ups:

- Add a `requirements.txt` that pins the Python demo dependencies
- Add automated tests that run easily on CI
- Convert the demo models into a packaged module with clearer I/O contracts

If you'd like, I can add a minimal `requirements.txt` and sample `.env.example` for you.

---

## License

MIT
