# 🧪 Epidemic Dynamics Simulator

> 📊 An interactive web-based simulator for modeling the dynamics of epidemic spread using nonlinear differential equations and real-time visualization.

---

## 🌟 Features

✅ Interactive UI to adjust 15 epidemiological parameters  
✅ Real-time plotting of system dynamics over time  
✅ Polar charts for 6 key time points  
✅ Nonlinear functions (`f(x) = ax³ + bx² + cx + d`) for parameter interactions  
✅ Time-dependent probabilities (`q1(t)`, `q2(t)`, etc.)  
✅ Randomize initial values with one click  
✅ Responsive design with Bootstrap 5

---

## 🧩 How It Works

The simulator solves a system of 15 coupled ordinary differential equations (ODEs) that model the evolution of key epidemic parameters:

- Mortality, infected population, hospitalizations, isolation, propagation speed, etc.
- Each parameter’s rate of change depends on nonlinear functions of other parameters.
- Time-varying probabilities simulate real-world fluctuations (e.g., seasonality, interventions).
- Results are visualized as **line graphs** and **polar charts**.

---

## 🛠️Stack

- **Python 3.8+**
- **Flask** — Web framework
- **Matplotlib** — Plotting
- **NumPy** — Numerical computations
- **SciPy** — ODE solver (`odeint`)
- **Bootstrap 5** — Responsive UI
- **Jinja2** — HTML templating

---

## 📁 Project Structure
```
epidemic-dynamics-simulator/
├── app.py 
├── requirements.txt 
├── README.md 
├── Profile 
├── static/
│ └── css/
│ └── styles.css 
└── templates/
└── index.html 
```
---

## 🚀 How to Run

1. **Install dependencies** using `requirements.txt`:
```bash
pip install -r requirements.txt
```
2. Start the server:
```bash
python app.py
```
3. Open your local server
   
---
## 📬 Автор
- DanilRose 👤
- totkto49@gmail.com 📧 

