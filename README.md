# AdaptIQ

**Hack-A-Thon: AI For Education - AI In Education**  
Organized by Unstop

**Team:** Bruteforce Army 

**Contributors:** 
- Saroj Sen(ML Model, Backend Integration)
- Tiyasa Saha(UI/UX Design & Frontend)
- Soumava Das(PPT Making & Research)

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

1. **Clone the repository:**  
	 `git clone https://github.com/sarojsenn/AdaptIQ.git`

2. **Install backend dependencies:**  
	 `cd AdaptIQ`  
	 `npm install` (for Node.js backend)  
	 or  
	 `pip install -r requirements.txt` (for Python backend)

3. **Start the backend server:**  
	 `npm start` or `python api_server.py`

4. **Open the frontend:**  
	 Open `client/pages/LandingPage.html` in your browser.

---

## Future Scopes

### 🟩 Dyslexia & Accessibility Support:
Dyslexia Support: Read-aloud mode, simplified fonts, color contrast adjustments.
Visually Impaired Learners: Screen reader compatibility, audio-based assessments.

---

### 📝 Extend subjects & update model:
Expand to multiple subjects, grade levels, and regional languages.
Also the model can be updated to 4pl; currently this is in 2pl.

---

### 🏫 Real-World Deployment:
Collaborate with schools, colleges, and ed-tech platforms to scale usage.

---

## 📄 License

MIT License

---
