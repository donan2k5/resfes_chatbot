from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import minimize
import logging

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

a = 1.0
c = 0.25  # 4 lựa chọn

def difficulty_to_b(level):
    mapping = {
        1: -2.5,   # Rất dễ
        2: -1.25,  # Dễ
        3: 0.0,    # Trung bình
        4: 1.25,   # Khó
        5: 2.5     # Rất khó
    }
    return mapping.get(level, 0.0)

def theta_to_level(theta):
    if theta < -2.0:
        return 1  # Rất dễ
    elif theta < -0.5:
        return 2  # Dễ
    elif theta < 0.5:
        return 3  # Trung bình
    elif theta < 2.0:
        return 4  # Khó
    else:
        return 5  # Rất khó

def P_theta(theta, a, b, c):
    exp_term = np.exp(-1.7 * a * (theta - b))
    return c + (1 - c) / (1 + exp_term)

# Hàm log-posterior cho MAP (log-likelihood + log-prior)
def log_posterior(theta, a_list, b_list, c_list, u_list):
    exp_term = np.exp(-1.7 * a_list * (theta - b_list))
    p_list = c_list + (1 - c_list) / (1 + exp_term)
    p_list = np.clip(p_list, 1e-6, 1 - 1e-6)
    ll = np.sum(u_list * np.log(p_list) + (1 - u_list) * np.log(1 - p_list))
    lp = -0.5 * theta ** 2
    return -(ll + lp)  # Đổi dấu để minimize

def estimate_theta_map(questions):
    a_list = np.array([a for _ in questions])
    b_list = np.array([difficulty_to_b(q['difficulty']) for q in questions])
    c_list = np.array([c for _ in questions])
    u_list = np.array([1 if q['isCorrect'] else 0 for q in questions])
    res = minimize(
        log_posterior,
        x0=0.0,
        args=(a_list, b_list, c_list, u_list),
        bounds=[(-3, 3)],
        method='L-BFGS-B'
    )
    return res.x[0]

@app.route('/calculate_theta', methods=['POST'])
def calculate_theta():
    try:
        data = request.get_json()
        logging.info(f"Received request data: {data}")
        questions = data
        theta = estimate_theta_map(questions)
        level = theta_to_level(theta)
        logging.info(f"Result: theta = {theta:.4f}, level = {level}")
        return jsonify({'theta': float(theta), 'level': int(level)})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8003)