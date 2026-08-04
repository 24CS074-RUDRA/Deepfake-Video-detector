<div align="center">

<!-- Header wave banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6D28D9,100:EC4899&height=220&section=header&text=DeepGuard%20AI&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Explainable%20Face-Swap%20Deepfake%20Detection%20System&descAlignY=55&descSize=18" width="100%"/>

<!-- Typing animation -->
<a href="#">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=3000&pause=800&color=A855F7&center=true&vCenter=true&width=650&lines=Detecting+Face-Swap+Deepfakes+with+AI+%F0%9F%94%8D;Multimodal+%7C+Explainable+%7C+Forensic-Grade;Built+for+Smart+India+Hackathon+2025+%F0%9F%87%AE%F0%9F%87%B3" alt="Typing SVG" />
</a>

<br/>

<!-- Badges -->
![SIH](https://img.shields.io/badge/SIH-Problem%20Statement%201683-orange?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgo=&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Made With](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F%20and%20Python-red?style=for-the-badge)

<br/>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

<br/>

![Stars](https://img.shields.io/github/stars/your-username/your-repo?style=social)
![Forks](https://img.shields.io/github/forks/your-username/your-repo?style=social)
![Issues](https://img.shields.io/github/issues/your-username/your-repo?color=blueviolet)

</div>

<br/>

<!-- Snake animation divider (enable via GitHub Action, see bottom) -->
<div align="center">
  <img src="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg" width="100%" alt="snake animation"/>
</div>

---

## 📌 Table of Contents

- [About The Project](#-about-the-project)
- [The Problem](#-the-problem)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Roadmap](#-roadmap)
- [Team](#-team)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About The Project

<img align="right" width="320" src="https://raw.githubusercontent.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/main/AI.gif" />

**DeepGuard AI** is an AI/ML-powered, **explainable** system built to detect **face-swap deepfake videos**.

Unlike conventional detectors that only output a binary *Real / Fake* label, DeepGuard AI produces a **complete forensic report** — highlighting manipulated regions, confidence scores, frequency anomalies, and audio-visual mismatches — so digital forensic experts, cybersecurity professionals, and law enforcement agencies can trust *and verify* the result.

> 🏆 Built for **Smart India Hackathon (SIH) 2025** — **Problem Statement SIH 1683**

<br clear="right"/>

## 🚨 The Problem

With the rise of Generative AI, face-swap deepfakes are increasingly weaponized for:

| 🎭 Misinformation | 🕵️ Identity Theft | 💰 Financial Fraud | 🏛️ Political Manipulation | 🔞 Non-Consensual Content |
|:---:|:---:|:---:|:---:|:---:|

As these fakes get more realistic, telling them apart from genuine footage has become a serious challenge — this project exists to close that gap.

---

## ⚙️ How It Works

```mermaid
flowchart LR
    A[🎥 Input Video] --> B[OpenCV + FFmpeg\nFrame & Audio Extraction]
    B --> C[RetinaFace\nFace Detection & Cropping]
    C --> D[EfficientNet\nSpatial Artifact Analysis]
    C --> E[BiLSTM\nTemporal Inconsistency]
    C --> F[FFT\nFrequency Domain Analysis]
    B --> G[Librosa + SyncNet\nAudio-Visual Sync Check]
    D --> H[🧠 Fusion Layer]
    E --> H
    F --> H
    G --> H
    H --> I[Grad-CAM + SHAP\nExplainability Layer]
    I --> J[📄 Forensic Report\nReal/Fake + Evidence]

    style A fill:#6D28D9,color:#fff
    style J fill:#EC4899,color:#fff
    style H fill:#111827,color:#fff
```

**Pipeline breakdown:**

1. 🎬 **Preprocessing** — Extract frames & audio using `OpenCV` + `FFmpeg`
2. 🙂 **Face Detection** — Crop precise facial regions with `RetinaFace`
3. 🔬 **Spatial Analysis** — Detect visual manipulation artifacts via `EfficientNet`
4. ⏱️ **Temporal Analysis** — Track frame-to-frame inconsistencies using `BiLSTM`
5. 🌊 **Frequency Analysis** — Uncover hidden GAN fingerprints using `FFT`
6. 🔊 **Audio-Visual Sync** — Verify lip-sync using `Librosa` + `SyncNet`
7. 🧩 **Fusion & Prediction** — Combine all modalities for a final verdict
8. 💡 **Explainability** — Visualize evidence using `Grad-CAM` + `SHAP`

---

## 🛠️ Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=py,pytorch,opencv,fastapi,react,docker,git,github&theme=dark" />

</div>

| Category | Technologies |
|---|---|
| **Language** | Python |
| **Computer Vision** | OpenCV, RetinaFace |
| **Deep Learning** | PyTorch, EfficientNet, BiLSTM |
| **Frequency Analysis** | Fast Fourier Transform (FFT) |
| **Audio Processing** | Librosa, SyncNet |
| **Explainable AI** | Grad-CAM, SHAP |
| **Backend** | FastAPI |
| **Frontend** | React |
| **Deployment** | Docker |

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Client
        UI[React Frontend]
    end
    subgraph Server
        API[FastAPI Backend]
        ML[ML Inference Engine]
        XAI[Explainability Module]
    end
    subgraph Infra
        DOCKER[Docker Containers]
    end

    UI -->|Upload Video| API
    API --> ML
    ML --> XAI
    XAI -->|Forensic Report| API
    API -->|JSON + Visual Evidence| UI
    API -.deployed via.-> DOCKER
    ML -.deployed via.-> DOCKER
```

---

## 📁 Project Structure

```bash
DeepGuard-AI/
├── 📂 backend/
│   ├── 📂 api/                # FastAPI routes
│   ├── 📂 models/              # PyTorch model definitions
│   ├── 📂 preprocessing/       # OpenCV / FFmpeg / RetinaFace pipeline
│   ├── 📂 explainability/      # Grad-CAM, SHAP modules
│   └── main.py
├── 📂 frontend/
│   ├── 📂 src/
│   └── package.json
├── 📂 dataset/                 # DFDC dataset scripts & loaders
├── 📂 notebooks/                # Colab / Jupyter experiments
├── 📂 docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
Node.js 18+
Docker & Docker Compose
CUDA-enabled GPU (recommended)
```

### Installation

<details>
<summary>📦 Click to expand setup instructions</summary>

```bash
# 1. Clone the repository
git clone https://github.com/your-username/DeepGuard-AI.git
cd DeepGuard-AI

# 2. Set up the backend
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Set up the frontend
cd ../frontend
npm install

# 4. Run with Docker (recommended)
docker-compose up --build
```

</details>

---

## ▶️ Usage

```bash
# Start backend API
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start frontend
cd frontend
npm start
```

Then open **`http://localhost:3000`**, upload a video, and get a full explainable forensic report 🕵️‍♂️

---

## 🗺️ Roadmap

- [x] Project proposal & problem statement finalized
- [x] Dataset pipeline (DFDC) setup in Google Colab
- [ ] Face & audio extraction (MTCNN, moviepy)
- [ ] Spatial model training (EfficientNet)
- [ ] Temporal model training (BiLSTM)
- [ ] Frequency-domain (FFT) module
- [ ] Audio-visual sync module (SyncNet)
- [ ] Explainability layer (Grad-CAM, SHAP)
- [ ] FastAPI backend integration
- [ ] React frontend dashboard
- [ ] Dockerized deployment
- [ ] Final SIH demo 🎉

---

## 👥 Team

<div align="center">


| Role | Name | ID / Enrollment |
| :--- | :--- | :--- |
| **Contributor** | **Rudra Patel** | 24CS074 |
| **Contributor** | **Varshil Patel** | 24CS080 |
| **Contributor** | **Trusha Patel** | 24CS078 |

</div>

---

## 🤝 Contributing

Contributions make the open-source community amazing — any contributions are **greatly appreciated**.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">

### ⭐ If you find this project useful, consider giving it a star!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:EC4899,100:6D28D9&height=150&section=footer" width="100%"/>

</div>
